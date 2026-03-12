"""
Reference Attention for cross-shot identity preservation (training-free).

Unlike intershot KV concat (which mixes reference tokens into self-attention
and conflicts with RoPE), Reference Attention computes a SEPARATE cross-attention
using the DiT's existing self-attention weights and adds the result as a
scaled residual.

Flow:
  Shot 0: generate normally, cache first-frame K,V (pre-RoPE) at last step
          using existing intershot cache mechanism.
  Shot N: at each denoising step, for layers in REF_ATTN_LAYERS:
          1. Normalize current hidden state with block.norm1
          2. Q = self_attn.norm_q(self_attn.q(normalized_x))  [no RoPE]
          3. K_ref, V_ref = cached from Shot 0  [no RoPE]
          4. out = flash_attention(Q, K_ref, V_ref)  [pure semantic matching]
          5. out = self_attn.o(out)
          6. x = x + alpha * out

Multi-Frame Extension (方案 6):
  Shot 0: cache K,V for multiple frames (first/mid/last) instead of just first.
  Shot N: Top-1 Routing — select the reference frame whose face yaw is closest
          to the current shot's keyframe face yaw.
"""

import numpy as np
import torch
from .modules.attention import flash_attention

# Default layers for reference attention (middle layers, identity-sensitive)
REF_ATTN_LAYERS = {15, 16, 17, 18, 19, 20}

# Default blending strength
DEFAULT_ALPHA = 0.3


def build_ref_attn_hook(block, ref_k, ref_v, alpha=DEFAULT_ALPHA):
    """Build a reference cross-attention hook for a specific block.

    Args:
        block: WanAttentionBlock instance (captures self_attn weights)
        ref_k: Cached reference K [B, L_ref, N, D], pre-RoPE, on CPU
        ref_v: Cached reference V [B, L_ref, N, D], on CPU
        alpha: Blending strength for the residual

    Returns:
        Callable hook: x -> x + alpha * ref_cross_attn(x, ref_k, ref_v)
    """
    sa = block.self_attn
    norm1 = block.norm1

    def hook(x):
        device = x.device
        b = x.size(0)
        n, d = sa.num_heads, sa.head_dim

        # Normalize current features (match self-attention input distribution)
        x_norm = norm1(x).to(x.dtype)

        # Compute Q from current state (no RoPE — pure semantic matching)
        q = sa.norm_q(sa.q(x_norm)).view(b, -1, n, d)

        # Move reference K,V to device
        k = ref_k.to(device=device, dtype=q.dtype)
        v = ref_v.to(device=device, dtype=q.dtype)

        # Separate reference cross-attention
        out = flash_attention(q, k, v)
        out = out.flatten(2)
        out = sa.o(out)

        # Scaled residual
        return x + alpha * out

    return hook


def build_ref_attn_config_for_model(ref_kv_cache, layers=None, alpha=DEFAULT_ALPHA):
    """Build ref_attn_config dict to pass to WanModel.forward().

    Args:
        ref_kv_cache: dict {layer_idx: (K, V)} from Shot 0's intershot cache.
            K shape: [B, L_ref, N, D], V shape: [B, L_ref, N, D]
        layers: set of layer indices (default: REF_ATTN_LAYERS)
        alpha: blending strength

    Returns:
        dict suitable for WanModel.forward(ref_attn_config=...)
    """
    if ref_kv_cache is None:
        return None
    return {
        'layers': layers or REF_ATTN_LAYERS,
        'kv_cache': ref_kv_cache,
        'alpha': alpha,
    }


# ---------------------------------------------------------------------------
# Multi-Frame Reference Attention (方案 6)
# ---------------------------------------------------------------------------


def detect_face_yaw(img_pil):
    """Detect face yaw angle from a PIL Image using InsightFace.

    Args:
        img_pil: PIL Image (RGB).

    Returns:
        float: Yaw angle in degrees, or 0.0 if no face detected.
    """
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    img_np = np.array(img_pil)[:, :, ::-1]  # RGB -> BGR
    faces = face_app.get(img_np)

    if not faces:
        print("  [WARN] No face detected for yaw estimation, defaulting to 0.0")
        return 0.0

    # Pick the largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    # face.pose returns [pitch, yaw, roll] in degrees
    yaw = float(face.pose[1])
    return yaw


def split_multiframe_kv_cache(kv_cache, n_frames):
    """Split concatenated multi-frame KV cache into per-frame caches.

    When cache_frame_indices has multiple frames, _extract_frame_kv concatenates
    all frame tokens. This function splits them back into individual frames.

    Args:
        kv_cache: dict {layer_idx: (K, V)}, where K/V shape [B, n_frames*H*W, N, D]
        n_frames: int, number of frames cached.

    Returns:
        list of n_frames dicts, each {layer_idx: (K, V)} with [B, H*W, N, D].
    """
    per_frame = [{} for _ in range(n_frames)]
    for layer_idx, (k, v) in kv_cache.items():
        total_tokens = k.size(1)
        hw = total_tokens // n_frames
        for fi in range(n_frames):
            start = fi * hw
            end = start + hw
            per_frame[fi][layer_idx] = (k[:, start:end], v[:, start:end])
    return per_frame


def select_ref_frame_by_yaw(ref_yaws, target_yaw):
    """Select the reference frame index with the closest face yaw.

    Args:
        ref_yaws: list of float, yaw angles for each reference frame.
        target_yaw: float, yaw angle for the target keyframe.

    Returns:
        int: Index of the closest reference frame.
    """
    diffs = [abs(y - target_yaw) for y in ref_yaws]
    best_idx = int(np.argmin(diffs))
    return best_idx


def extract_ref_frame_images(video_tensor, frame_indices, resize=None):
    """Extract specific frame images from a video tensor.

    Args:
        video_tensor: Tensor (C, T, H, W) in [-1, 1].
        frame_indices: list of temporal frame indices (in pixel space, not latent).
        resize: optional (W, H) tuple for resizing.

    Returns:
        list of PIL Images.
    """
    from PIL import Image

    frames = []
    for idx in frame_indices:
        if idx >= video_tensor.shape[1]:
            idx = video_tensor.shape[1] - 1
        frame = video_tensor[:, idx, :, :]  # (C, H, W)
        frame = ((frame.clamp(-1, 1) + 1) / 2 * 255).byte()
        frame = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C) RGB
        img = Image.fromarray(frame)
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        frames.append(img)
    return frames
