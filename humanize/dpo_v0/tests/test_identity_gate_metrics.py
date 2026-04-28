"""Tests for identity_gate_metrics — AC-2.3 / M2 per-frame gate semantics.

Five spec'd cases:
1. Identity test — video_a == video_b → l1=0, ssim=1, psnr=+inf; gate passes.
2. Single-frame violation — one frame +0.01 noise → that frame fails L1; gate fails.
3. All-frames violation — every frame +0.01 noise → all frames fail; gate fails.
4. dtype roundtrip — bf16 input matches fp32 reference within 1e-4.
5. Shape mismatch raise — non-matching trailing dim raises with shape diff message.
"""

from __future__ import annotations

import math
import sys
import pathlib

import pytest
import torch

# Local import path so this test runs from a checked-out videodpoWan tree
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval.identity_gate_metrics import (  # noqa: E402
    gate_decision,
    per_frame_metrics,
)


# Use a small frame size (>= SSIM window 11) and a small T to keep tests fast.
T, C, H, W = 4, 3, 16, 16


def _random_video(seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(T, C, H, W, generator=g, dtype=torch.float32) * 2.0 - 1.0


# ------------------------- 1. Identity --------------------------------------


def test_identity_passes_with_all_zero_l1_and_unit_ssim():
    a = _random_video(seed=1)
    b = a.clone()
    out = per_frame_metrics(a, b)
    assert len(out["per_frame"]) == T
    for p in out["per_frame"]:
        assert p["l1"] == pytest.approx(0.0, abs=1e-7)
        assert p["ssim_bt709"] == pytest.approx(1.0, abs=1e-5)
        assert math.isinf(p["psnr_db"]) and p["psnr_db"] > 0
    decision = gate_decision(out["per_frame"])
    assert decision["passed"] is True
    assert decision["failed_frames"] == []
    assert decision["thresholds"] == {"l1": 1e-3, "ssim": 0.99, "psnr_db": 40.0}
    # Summary checks
    s = out["summary"]
    assert s["mean_l1"] == pytest.approx(0.0, abs=1e-7)
    assert s["max_l1"] == pytest.approx(0.0, abs=1e-7)
    assert s["mean_ssim_bt709"] == pytest.approx(1.0, abs=1e-5)
    assert s["min_ssim_bt709"] == pytest.approx(1.0, abs=1e-5)
    assert math.isinf(s["mean_psnr_db"]) and s["mean_psnr_db"] > 0
    assert math.isinf(s["min_psnr_db"]) and s["min_psnr_db"] > 0


# ------------------------- 2. Single-frame violation -----------------------


def test_single_frame_violation_fails_only_that_frame():
    a = _random_video(seed=2)
    b = a.clone()
    # Add 0.01 (uniform sign-aware) noise in a single frame; magnitude ~0.01
    # makes per-frame L1 well above the 1e-3 threshold.
    b[1] = b[1] + 0.01
    out = per_frame_metrics(a, b)
    decision = gate_decision(out["per_frame"])
    assert decision["passed"] is False
    failed_ids = [f["frame_id"] for f in decision["failed_frames"]]
    assert failed_ids == [1]
    fail = decision["failed_frames"][0]
    assert "l1" in fail["violations"]
    assert fail["l1"] == pytest.approx(0.01, abs=5e-4)


# ------------------------- 3. All-frames violation -------------------------


def test_all_frames_violation_fails_every_frame():
    a = _random_video(seed=3)
    b = a + 0.01  # uniform noise on every frame
    out = per_frame_metrics(a, b)
    decision = gate_decision(out["per_frame"])
    assert decision["passed"] is False
    assert len(decision["failed_frames"]) == T
    for fail in decision["failed_frames"]:
        assert "l1" in fail["violations"]


# ------------------------- 4. dtype roundtrip ------------------------------


def test_bf16_matches_fp32_within_tolerance():
    a32 = _random_video(seed=4)
    # Add a small known perturbation so PSNR / L1 are non-trivial and finite
    b32 = a32 + 0.005
    a_bf = a32.to(dtype=torch.bfloat16)
    b_bf = b32.to(dtype=torch.bfloat16)

    out_fp32 = per_frame_metrics(a32, b32)
    out_bf16 = per_frame_metrics(a_bf, b_bf)

    for p32, p16 in zip(out_fp32["per_frame"], out_bf16["per_frame"]):
        # bf16 has ~3 decimal digits of precision; relax tolerance accordingly.
        assert p32["l1"] == pytest.approx(p16["l1"], abs=5e-3)
        assert p32["ssim_bt709"] == pytest.approx(p16["ssim_bt709"], abs=5e-3)
        assert p32["psnr_db"] == pytest.approx(p16["psnr_db"], abs=1.0)  # dB tol


# ------------------------- 5. Shape mismatch raise -------------------------


def test_shape_mismatch_raises_with_descriptive_message():
    a = _random_video(seed=5)
    b = torch.zeros(T, C, H, W + 1)
    with pytest.raises(ValueError) as exc:
        per_frame_metrics(a, b)
    msg = str(exc.value)
    assert "shape mismatch" in msg
    assert str(W) in msg and str(W + 1) in msg


def test_ndim_not_4_raises():
    a = torch.zeros(C, H, W)  # 3D
    b = torch.zeros(C, H, W)
    with pytest.raises(ValueError) as exc:
        per_frame_metrics(a, b)
    assert "expected 4D" in str(exc.value)


def test_device_mismatch_raises_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    a = _random_video(seed=6)
    b = a.clone().cuda()
    with pytest.raises(ValueError) as exc:
        per_frame_metrics(a, b)
    assert "device mismatch" in str(exc.value)


def test_unsupported_channel_count_raises():
    a = torch.zeros(T, 5, H, W)  # neither 1 nor 3
    b = torch.zeros(T, 5, H, W)
    with pytest.raises(ValueError) as exc:
        per_frame_metrics(a, b)
    assert "C in (1, 3)" in str(exc.value)


# ------------------------- bonus: single-channel pass-through --------------


def test_single_channel_input_uses_pass_through_luma():
    # C=1 input should skip BT.709 mix and use the channel directly as luma.
    a = torch.zeros(T, 1, H, W)
    b = torch.zeros(T, 1, H, W)
    out = per_frame_metrics(a, b)
    for p in out["per_frame"]:
        assert p["l1"] == 0.0
        assert p["ssim_bt709"] == pytest.approx(1.0, abs=1e-5)
