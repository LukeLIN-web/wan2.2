"""Pred-x0 Feature Injection: soft face mask computation in DiT token space.

Detects face in keyframe image, maps bbox to token-space coordinates with
Gaussian falloff mask. Returns cache/inject indices for the denoising loop.
"""

import math

import numpy as np
import torch

from .utils.utils import best_output_size


def compute_predx0_face_mask(
    img_pil, wan_model, frame_num=81, max_area=704 * 1280,
    expand_ratio=2.5, pad_tokens=2, sigma_scale=0.5,
):
    """Detect face and compute soft Gaussian mask in DiT token space.

    Handles image preprocessing (resize + center crop) identically to i2v.

    Args:
        img_pil: PIL Image (RGB) — the keyframe.
        wan_model: WanTI2V instance (provides vae_stride, patch_size).
        frame_num: int, number of video frames (default 81).
        max_area: int, max pixel area (default 704*1280).
        expand_ratio: float, face bbox expansion (default 2.5).
        pad_tokens: int, padding around token bbox for soft boundary.
        sigma_scale: float, Gaussian sigma = max(bbox_w, bbox_h) * sigma_scale.

    Returns:
        cache_indices: LongTensor [N], frame-0 face token indices.
        inject_indices: LongTensor [N*(F_lat-1)], non-first-frame indices.
        inject_weights: FloatTensor [N*(F_lat-1)], Gaussian weights.
        bbox_info: dict with detection metadata.
    """
    ih, iw = img_pil.height, img_pil.width
    dh = wan_model.patch_size[1] * wan_model.vae_stride[1]
    dw = wan_model.patch_size[2] * wan_model.vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

    # Resize + center crop offsets (same as i2v)
    scale = max(ow / iw, oh / ih)
    resized_w, resized_h = round(iw * scale), round(ih * scale)
    cx1 = (resized_w - ow) // 2
    cy1 = (resized_h - oh) // 2

    # Detect face
    bbox_pixel, det_method = _detect_face_bbox(img_pil, expand_ratio)
    bx1, by1, bx2, by2 = bbox_pixel

    # Map bbox: original → resized → center-cropped
    bx1 = max(0, bx1 * scale - cx1)
    by1 = max(0, by1 * scale - cy1)
    bx2 = min(ow, bx2 * scale - cx1)
    by2 = min(oh, by2 * scale - cy1)

    # Token grid dimensions
    token_stride_h = wan_model.vae_stride[1] * wan_model.patch_size[1]
    token_stride_w = wan_model.vae_stride[2] * wan_model.patch_size[2]
    H_tokens = oh // token_stride_h
    W_tokens = ow // token_stride_w
    F_lat = (frame_num - 1) // wan_model.vae_stride[0] + 1
    hw_tokens = H_tokens * W_tokens

    # Token-space bbox
    t_y1 = int(by1) // token_stride_h
    t_x1 = int(bx1) // token_stride_w
    t_y2 = min(H_tokens, math.ceil(by2 / token_stride_h))
    t_x2 = min(W_tokens, math.ceil(bx2 / token_stride_w))

    # Padded bbox for soft mask
    pt_y1 = max(0, t_y1 - pad_tokens)
    pt_x1 = max(0, t_x1 - pad_tokens)
    pt_y2 = min(H_tokens, t_y2 + pad_tokens)
    pt_x2 = min(W_tokens, t_x2 + pad_tokens)

    # Gaussian parameters
    tc_y = (t_y1 + t_y2) / 2
    tc_x = (t_x1 + t_x2) / 2
    sigma = max(t_x2 - t_x1, t_y2 - t_y1, 1) * sigma_scale

    # Build spatial indices and Gaussian weights
    spatial_indices = []
    weights = []
    for r in range(pt_y1, pt_y2):
        for c in range(pt_x1, pt_x2):
            spatial_indices.append(r * W_tokens + c)
            dist = math.sqrt((c + 0.5 - tc_x) ** 2 + (r + 0.5 - tc_y) ** 2)
            weights.append(math.exp(-dist ** 2 / (2 * sigma ** 2)))

    if not spatial_indices:
        raise ValueError(
            f"Face bbox maps to zero tokens. Pixel bbox: "
            f"({bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f}), "
            f"token grid: {H_tokens}x{W_tokens}"
        )

    spatial_indices = torch.tensor(spatial_indices, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.max()  # normalize to [0, 1]

    # Cache indices: frame 0 face tokens
    cache_indices = spatial_indices.clone()

    # Inject indices: frames 1..F_lat-1
    inject_idx_parts = []
    inject_w_parts = []
    for k in range(1, F_lat):
        inject_idx_parts.append(spatial_indices + k * hw_tokens)
        inject_w_parts.append(weights.clone())

    inject_indices = torch.cat(inject_idx_parts)
    inject_weights = torch.cat(inject_w_parts)

    n_face = len(spatial_indices)
    bbox_info = {
        "pixel_bbox": list(bbox_pixel),
        "token_bbox": [t_x1, t_y1, t_x2, t_y2],
        "padded_token_bbox": [pt_x1, pt_y1, pt_x2, pt_y2],
        "num_spatial_tokens": n_face,
        "detection_method": det_method,
        "H_tokens": H_tokens,
        "W_tokens": W_tokens,
        "F_lat": F_lat,
        "hw_tokens": hw_tokens,
    }

    print(
        f"  PredX0 mask: {n_face} spatial tokens, "
        f"bbox ({t_x1},{t_y1})-({t_x2},{t_y2}), "
        f"padded ({pt_x1},{pt_y1})-({pt_x2},{pt_y2}), "
        f"inject {F_lat - 1} frames = {len(inject_indices)} tokens"
    )
    return cache_indices, inject_indices, inject_weights, bbox_info


def _detect_face_bbox(img_pil, expand_ratio=2.5):
    """Detect face and return expanded bbox (same logic as anchor_kv)."""
    try:
        from insightface.app import FaceAnalysis

        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        img_np = np.array(img_pil)[:, :, ::-1]
        faces = face_app.get(img_np)

        if faces:
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            x1, y1, x2, y2 = face.bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            crop_h = h * expand_ratio
            crop_w = crop_h
            left = max(0, int(cx - crop_w / 2))
            top = max(0, int(cy - crop_h * 0.3))
            right = min(img_pil.width, int(cx + crop_w / 2))
            bottom = min(img_pil.height, int(top + crop_h))
            return (left, top, right, bottom), "insightface"

    except Exception as e:
        print(f"  [WARN] InsightFace failed: {e}")

    # Fallback: center crop
    w_img, h_img = img_pil.size
    crop_size = int(min(w_img, h_img) * 0.7)
    left = (w_img - crop_size) // 2
    top = max(0, (h_img - crop_size) // 2 - int(crop_size * 0.1))
    return (left, top, left + crop_size, top + crop_size), "center_crop_fallback"
