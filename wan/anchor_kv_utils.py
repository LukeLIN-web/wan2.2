"""
Utility for extracting identity-preserving KV cache from a cropped anchor
region (face + upper body) of an image.

Usage:
    kv = extract_anchor_kv(wan_model, img, bbox_pixel, layers={15,16,17,18,19,20})
    # kv: {layer_idx: (K_cpu, V_cpu)}  — K is pre-RoPE
"""

import gc
import math
from contextlib import contextmanager

import torch
import torchvision.transforms.functional as TF

from .utils.utils import best_output_size, masks_like


def extract_anchor_kv(
    wan_model,
    img,
    bbox_pixel,
    layers,
    frame_num=81,
    max_area=704 * 1280,
    offload_model=True,
):
    """
    Extract identity-preserving KV cache from a cropped anchor region.

    Args:
        wan_model: A WanTI2V instance (from wan.textimage2video).
        img: PIL Image (full anchor frame, RGB).
        bbox_pixel: tuple (left, top, right, bottom) in original pixel coords
            of the person region to extract.
        layers: set of DiT layer indices to extract KV from
            (e.g. {15, 16, 17, 18, 19, 20}).
        frame_num: int, number of video frames (default 81).
        max_area: int, max pixel area (default 704*1280).
        offload_model: bool, whether to offload model after extraction.

    Returns:
        dict: {layer_idx: (K, V)} where K, V are CPU tensors containing ONLY
            tokens inside the bbox region.  K is pre-RoPE (for TcRoPE
            application later).
    """
    device = wan_model.device

    # ------------------------------------------------------------------ #
    # 1. Resize + center-crop (mirrors WanTI2V.i2v lines 516-528)
    # ------------------------------------------------------------------ #
    ih, iw = img.height, img.width
    dh = wan_model.patch_size[1] * wan_model.vae_stride[1]
    dw = wan_model.patch_size[2] * wan_model.vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

    scale = max(ow / iw, oh / ih)
    img_resized = img.resize(
        (round(iw * scale), round(ih * scale)),
        resample=3,  # PIL.Image.LANCZOS
    )

    # center-crop offsets (in resized-image coords)
    cx1 = (img_resized.width - ow) // 2
    cy1 = (img_resized.height - oh) // 2
    img_cropped = img_resized.crop((cx1, cy1, cx1 + ow, cy1 + oh))
    assert img_cropped.width == ow and img_cropped.height == oh

    # ------------------------------------------------------------------ #
    # 2. Map bbox from original pixel coords to output-size coords
    # ------------------------------------------------------------------ #
    bx1, by1, bx2, by2 = bbox_pixel
    # Transform: original -> resized -> cropped
    bx1 = bx1 * scale - cx1
    by1 = by1 * scale - cy1
    bx2 = bx2 * scale - cx1
    by2 = by2 * scale - cy1

    # Clamp to valid range
    bx1 = max(0, bx1)
    by1 = max(0, by1)
    bx2 = min(ow, bx2)
    by2 = min(oh, by2)

    # ------------------------------------------------------------------ #
    # 3. Convert pixel bbox to latent-space token coords
    # ------------------------------------------------------------------ #
    token_stride_h = wan_model.vae_stride[1] * wan_model.patch_size[1]  # 32
    token_stride_w = wan_model.vae_stride[2] * wan_model.patch_size[2]  # 32

    t_y1 = int(by1) // token_stride_h
    t_x1 = int(bx1) // token_stride_w
    t_y2 = math.ceil(by2 / token_stride_h)
    t_x2 = math.ceil(bx2 / token_stride_w)

    H_tokens = oh // token_stride_h
    W_tokens = ow // token_stride_w

    # Build flat index list for tokens inside the bbox (row-major)
    bbox_indices = []
    for r in range(t_y1, t_y2):
        for c in range(t_x1, t_x2):
            bbox_indices.append(r * W_tokens + c)

    if len(bbox_indices) == 0:
        raise ValueError(
            f"Bounding box maps to zero tokens. "
            f"Pixel bbox (mapped): ({bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}), "
            f"Token grid: {H_tokens}x{W_tokens}"
        )

    bbox_indices = torch.tensor(bbox_indices, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # 4. VAE encode the image
    # ------------------------------------------------------------------ #
    img_tensor = (
        TF.to_tensor(img_cropped)
        .sub_(0.5)
        .div_(0.5)
        .to(device)
        .unsqueeze(1)
    )  # [3, 1, oh, ow]

    z = wan_model.vae.encode([img_tensor])  # list of [C, T_lat, H_lat, W_lat]

    # ------------------------------------------------------------------ #
    # 5. Build latent for DiT forward
    # ------------------------------------------------------------------ #
    F = frame_num
    F_lat = (F - 1) // wan_model.vae_stride[0] + 1
    H_lat = oh // wan_model.vae_stride[1]
    W_lat = ow // wan_model.vae_stride[2]

    # For KV extraction we want the model to see the clean image at t=0.
    # Build noise of same shape as z but we will use t=0 so noise doesn't matter.
    noise = torch.zeros(
        wan_model.vae.model.z_dim, F_lat, H_lat, W_lat,
        dtype=torch.float32, device=device,
    )

    # Mask: same construction as i2v — first frame clean, rest noisy
    mask1, mask2 = masks_like([noise], zero=True)
    latent = (1.0 - mask2[0]) * z[0] + mask2[0] * noise

    seq_len = int(
        math.ceil(
            F_lat * H_lat * W_lat
            / (wan_model.patch_size[1] * wan_model.patch_size[2])
        )
    )
    seq_len = int(math.ceil(seq_len / wan_model.sp_size)) * wan_model.sp_size

    # ------------------------------------------------------------------ #
    # 6. T5 encode empty prompt (we only want visual KV)
    # ------------------------------------------------------------------ #
    if not wan_model.t5_cpu:
        wan_model.text_encoder.model.to(device)
        context = wan_model.text_encoder([""], device)
        if offload_model:
            wan_model.text_encoder.model.cpu()
    else:
        context = wan_model.text_encoder([""], torch.device("cpu"))
        context = [t.to(device) for t in context]

    # ------------------------------------------------------------------ #
    # 7. Single DiT forward pass with cache_kv=True
    # ------------------------------------------------------------------ #
    if offload_model or wan_model.init_on_cpu:
        wan_model.model.to(device)
        torch.cuda.empty_cache()

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(wan_model.model, "no_sync", noop_no_sync)

    with (
        torch.amp.autocast("cuda", dtype=wan_model.param_dtype),
        torch.no_grad(),
        no_sync(),
    ):
        latent_model_input = [latent.to(device)]

        # t=0 means the model sees a clean (fully denoised) sample.
        # Build the per-token timestep exactly like i2v does.
        t_val = torch.zeros(1, device=device)
        temp_ts = (mask2[0][0][:, ::2, ::2] * t_val).flatten()
        temp_ts = torch.cat([
            temp_ts,
            temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_val,
        ])
        timestep = temp_ts.unsqueeze(0)

        result = wan_model.model(
            latent_model_input,
            t=timestep,
            context=[context[0]],
            seq_len=seq_len,
            cache_kv=True,
            cache_frame_indices=[0],  # first frame only
            intershot_layers=layers,
            cache_strip_rope=True,  # pre-RoPE K for TcRoPE
        )

        # result = (output_list, kv_cache_dict)
        raw_cache = result[1]  # {layer_idx: (K, V)}

    # ------------------------------------------------------------------ #
    # 8. Filter KV tokens by bbox
    # ------------------------------------------------------------------ #
    filtered_kv = {}
    for layer_idx, (k, v) in raw_cache.items():
        # k, v shape: [num_heads, num_tokens, head_dim] or [B, num_heads, num_tokens, head_dim]
        # cache_frame_indices=[0] means we only cached the first frame's tokens,
        # so num_tokens == H_tokens * W_tokens (spatial tokens of frame 0).
        filtered_kv[layer_idx] = (
            k[:, bbox_indices].cpu(),
            v[:, bbox_indices].cpu(),
        )

    # ------------------------------------------------------------------ #
    # 9. Cleanup
    # ------------------------------------------------------------------ #
    del raw_cache, result, latent, noise, z, img_tensor, context
    del latent_model_input, timestep, temp_ts

    if offload_model:
        wan_model.model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    return filtered_kv
