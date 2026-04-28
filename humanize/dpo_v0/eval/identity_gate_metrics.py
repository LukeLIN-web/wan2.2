"""Identity-gate per-frame metrics + gate decision for AC-2.3 / M2.

Two video tensors of shape (T, C, H, W) in input range [-1, 1] are compared
frame-by-frame via L1, BT.709-luma SSIM, and PSNR. The per-frame gate is the
pass criterion (every frame must satisfy all three thresholds); mean-over-
frames stats are reporting only.

Conventions (do not change without DEC-CAL sign-off):
- Input range is [-1, 1]; metrics are computed in this range (no rescale to
  [0, 1]). Implies SSIM/PSNR dynamic range L = 2.0.
- Luma is BT.709: ``Y' = 0.2126*R + 0.7152*G + 0.0722*B``.
- SSIM window is an 11x11 Gaussian (sigma=1.5), K1=0.01, K2=0.03.
- PSNR is reported in dB; identical frames return ``+inf``.
- Compute is fp32 regardless of input dtype (bf16/fp16/fp32 accepted).

Default thresholds match AC-2.3 preliminary calibration:
``l1=1e-3, ssim=0.99, psnr_db=40.0``. Caller may pass overrides; per the
DEC-CAL framing the trainer/manifest layer is responsible for any reviewer
sign-off, not this module.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


# BT.709 luma weights (Rec. ITU-R BT.709-6 Annex 1).
_BT709_R = 0.2126
_BT709_G = 0.7152
_BT709_B = 0.0722

# Input dynamic range. Inputs are [-1, 1] so peak-to-peak L = 2.
_DYNAMIC_RANGE = 2.0
_K1 = 0.01
_K2 = 0.03
_C1 = (_K1 * _DYNAMIC_RANGE) ** 2
_C2 = (_K2 * _DYNAMIC_RANGE) ** 2

_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5


def _validate_pair(video_a: torch.Tensor, video_b: torch.Tensor) -> None:
    if not isinstance(video_a, torch.Tensor) or not isinstance(video_b, torch.Tensor):
        raise TypeError(
            f"video_a / video_b must be torch.Tensor; got {type(video_a)} / {type(video_b)}"
        )
    if video_a.shape != video_b.shape:
        raise ValueError(
            f"shape mismatch: video_a.shape={tuple(video_a.shape)} "
            f"video_b.shape={tuple(video_b.shape)}"
        )
    if video_a.ndim != 4:
        raise ValueError(
            f"expected 4D tensor (T, C, H, W); got ndim={video_a.ndim} "
            f"shape={tuple(video_a.shape)}"
        )
    T = video_a.shape[0]
    if T < 1:
        raise ValueError(f"T must be >= 1; got T={T}")
    if video_a.device != video_b.device:
        raise ValueError(
            f"device mismatch: video_a.device={video_a.device} "
            f"video_b.device={video_b.device}"
        )


def _to_fp32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype is torch.float32:
        return x
    return x.to(dtype=torch.float32)


def _bt709_luma(rgb: torch.Tensor) -> torch.Tensor:
    """Convert (T, 3, H, W) RGB in [-1, 1] to (T, 1, H, W) BT.709 luma."""
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    return _BT709_R * r + _BT709_G * g + _BT709_B * b


def _gaussian_window_2d(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] @ g1d[None, :]
    return g2d.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)


def _ssim_per_frame(luma_a: torch.Tensor, luma_b: torch.Tensor) -> torch.Tensor:
    """Per-frame SSIM (window-mean) on (T, 1, H, W) single-channel input.

    Returns a tensor of shape (T,) with per-frame SSIM in fp32.
    """
    if luma_a.shape != luma_b.shape:
        raise ValueError("luma_a / luma_b shape mismatch")
    T, C, H, W = luma_a.shape
    if C != 1:
        raise ValueError(f"expected single-channel luma; got C={C}")

    pad = _SSIM_WINDOW // 2
    if H < _SSIM_WINDOW or W < _SSIM_WINDOW:
        raise ValueError(
            f"frame too small for SSIM window={_SSIM_WINDOW}: H={H} W={W}"
        )

    window = _gaussian_window_2d(_SSIM_WINDOW, _SSIM_SIGMA, luma_a.device, luma_a.dtype)

    mu_a = F.conv2d(luma_a, window, padding=pad)
    mu_b = F.conv2d(luma_b, window, padding=pad)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = F.conv2d(luma_a * luma_a, window, padding=pad) - mu_a2
    sigma_b2 = F.conv2d(luma_b * luma_b, window, padding=pad) - mu_b2
    sigma_ab = F.conv2d(luma_a * luma_b, window, padding=pad) - mu_ab

    numerator = (2 * mu_ab + _C1) * (2 * sigma_ab + _C2)
    denominator = (mu_a2 + mu_b2 + _C1) * (sigma_a2 + sigma_b2 + _C2)
    ssim_map = numerator / denominator
    # Clamp into [-1, 1] to absorb any tiny numerical excursion above 1.0
    # caused by conv1d(a*a) - mu_a^2 going slightly negative under fp32.
    ssim_map = ssim_map.clamp(-1.0, 1.0)
    return ssim_map.flatten(start_dim=1).mean(dim=1)


def per_frame_metrics(video_a: torch.Tensor, video_b: torch.Tensor) -> dict[str, Any]:
    """Compute per-frame L1 / SSIM (BT.709 luma) / PSNR + mean-over-frames summary.

    Args:
        video_a, video_b: (T, C, H, W) in [-1, 1]; bf16/fp16/fp32 accepted; same
            device, same shape. C must be 1 or 3 (luma path only valid for 3).

    Returns:
        {
            "per_frame": [{"frame_id": int, "l1": float, "ssim_bt709": float,
                           "psnr_db": float}, ...],
            "summary": {
                "mean_l1": float, "max_l1": float,
                "mean_ssim_bt709": float, "min_ssim_bt709": float,
                "mean_psnr_db": float, "min_psnr_db": float,
            },
        }
    """
    _validate_pair(video_a, video_b)
    if video_a.shape[1] not in (1, 3):
        raise ValueError(
            f"expected C in (1, 3) for luma conversion; got C={video_a.shape[1]}"
        )

    a = _to_fp32(video_a)
    b = _to_fp32(video_b)

    # L1 per frame: mean |a - b| over (C, H, W)
    diff = (a - b).abs()
    l1_per_frame = diff.flatten(start_dim=1).mean(dim=1)  # (T,)

    # PSNR per frame: 10 * log10(MAX^2 / MSE); identical frames -> +inf
    mse_per_frame = ((a - b) ** 2).flatten(start_dim=1).mean(dim=1)  # (T,)
    psnr_per_frame = torch.empty_like(mse_per_frame)
    zero_mask = mse_per_frame == 0
    nonzero_mask = ~zero_mask
    if nonzero_mask.any():
        psnr_per_frame[nonzero_mask] = 10.0 * torch.log10(
            (_DYNAMIC_RANGE ** 2) / mse_per_frame[nonzero_mask]
        )
    psnr_per_frame[zero_mask] = float("inf")

    # SSIM per frame on BT.709 luma (or pass-through if C==1)
    if a.shape[1] == 3:
        luma_a = _bt709_luma(a)
        luma_b = _bt709_luma(b)
    else:
        luma_a = a
        luma_b = b
    ssim_per_frame = _ssim_per_frame(luma_a, luma_b)  # (T,)

    T = a.shape[0]
    per_frame = []
    for i in range(T):
        per_frame.append(
            {
                "frame_id": i,
                "l1": float(l1_per_frame[i].item()),
                "ssim_bt709": float(ssim_per_frame[i].item()),
                "psnr_db": float(psnr_per_frame[i].item()),
            }
        )

    l1_vals = [p["l1"] for p in per_frame]
    ssim_vals = [p["ssim_bt709"] for p in per_frame]
    psnr_vals = [p["psnr_db"] for p in per_frame]

    # mean_psnr_db semantics:
    #   - all frames identical → +inf
    #   - mixed (some identical, some not) → mean of the FINITE frames only,
    #     so a single non-identical frame gives a meaningful mean rather than
    #     being swallowed into the +inf from identical frames.
    #   - all frames non-identical → arithmetic mean of all frames.
    # Callers stamping into manifests should label this field
    # "finite_mean_psnr_db" or document the semantics explicitly.
    if all(math.isinf(v) for v in psnr_vals):
        mean_psnr = float("inf")
    else:
        finite = [v for v in psnr_vals if math.isfinite(v)]
        mean_psnr = sum(finite) / len(finite)

    summary = {
        "mean_l1": sum(l1_vals) / T,
        "max_l1": max(l1_vals),
        "mean_ssim_bt709": sum(ssim_vals) / T,
        "min_ssim_bt709": min(ssim_vals),
        "mean_psnr_db": mean_psnr,
        "min_psnr_db": min(psnr_vals),
    }
    return {"per_frame": per_frame, "summary": summary}


def gate_decision(
    per_frame: list[dict[str, Any]],
    l1_threshold: float = 1e-3,
    ssim_threshold: float = 0.99,
    psnr_threshold_db: float = 40.0,
) -> dict[str, Any]:
    """Per-frame gate decision. Pass = ALL frames satisfy ALL three thresholds.

    Args:
        per_frame: output of ``per_frame_metrics(...)["per_frame"]``.
        l1_threshold: per-frame L1 strictly less than this passes.
        ssim_threshold: per-frame SSIM at least this passes.
        psnr_threshold_db: per-frame PSNR (dB) at least this passes.

    Returns:
        {
            "passed": bool,
            "failed_frames": [
                {"frame_id": int, "l1": float, "ssim_bt709": float,
                 "psnr_db": float, "violations": ["l1" | "ssim" | "psnr", ...]},
                ...,
            ],
            "thresholds": {"l1": float, "ssim": float, "psnr_db": float},
        }
    """
    failed: list[dict[str, Any]] = []
    for p in per_frame:
        violations = []
        if p["l1"] >= l1_threshold:
            violations.append("l1")
        if p["ssim_bt709"] < ssim_threshold:
            violations.append("ssim")
        if p["psnr_db"] < psnr_threshold_db:
            violations.append("psnr")
        if violations:
            failed.append(
                {
                    "frame_id": p["frame_id"],
                    "l1": p["l1"],
                    "ssim_bt709": p["ssim_bt709"],
                    "psnr_db": p["psnr_db"],
                    "violations": violations,
                }
            )
    return {
        "passed": len(failed) == 0,
        "failed_frames": failed,
        "thresholds": {
            "l1": l1_threshold,
            "ssim": ssim_threshold,
            "psnr_db": psnr_threshold_db,
        },
    }
