"""Watchdog unit tests.

Run from the videodpoWan repo root::

    cd /shared/user60/worldmodel/rlvideo/videodpoWan
    python -m pytest humanize/dpo_v0/watchdog/tests/ -v

Tests are pure-Python and do NOT require CUDA. CUDA-dependent paths in
``vram_probe`` are gated by ``torch.cuda.is_available()``; tests
exercise the no-CUDA fallback (no records written) and the JSONL
plumbing under a stubbed-stage path.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent  # humanize/dpo_v0/
sys.path.insert(0, str(PKG_ROOT.parent.parent))  # videodpoWan
sys.path.insert(0, str(PKG_ROOT))

from watchdog.loss_components import KNOWN_COMPONENT_FIELDS, LossComponentLogger
from watchdog.routing_counter import (  # noqa: E402
    LowNoiseRoutingError,
    RoutingCounterTail,
    detect_expert,
)
from watchdog.vram_probe import VramProbe  # noqa: E402
from watchdog import Watchdog  # noqa: E402


# ---------- routing_counter ----------


def test_detect_expert_boundary():
    assert detect_expert(901) == "high_noise"
    assert detect_expert(900) == "low_noise"
    assert detect_expert(999) == "high_noise"
    assert detect_expert(0) == "low_noise"


def test_routing_tail_high_noise_increments(tmp_path: pathlib.Path):
    tail = RoutingCounterTail(out_path=tmp_path / "routing.jsonl", halt_on_low_noise=True)
    tail.log(step=0, sampled_timestep_id=0, raw_timestep=950, pair_id="p0")
    tail.log(step=1, sampled_timestep_id=1, raw_timestep=999, pair_id="p1")
    assert tail.high_count == 2
    assert tail.low_count == 0
    summary = tail.summary()
    assert summary["fraction_high_noise"] == 1.0
    assert summary["total_forwards"] == 2

    lines = (tmp_path / "routing.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[1])
    assert rec1["cum_high"] == 2
    assert rec1["cum_low"] == 0
    assert rec1["detected_expert"] == "high_noise"


def test_routing_tail_halts_on_low_noise(tmp_path: pathlib.Path):
    tail = RoutingCounterTail(out_path=tmp_path / "routing.jsonl", halt_on_low_noise=True)
    with pytest.raises(LowNoiseRoutingError):
        tail.log(step=0, sampled_timestep_id=0, raw_timestep=500, pair_id="p0")
    # Even though the call raised, the JSONL line and the counter must
    # both have been written before the raise so post-mortem can read
    # the offending event.
    assert tail.low_count == 1
    rec = json.loads((tmp_path / "routing.jsonl").read_text().splitlines()[0])
    assert rec["detected_expert"] == "low_noise"


def test_routing_tail_can_record_low_noise_when_halt_disabled(tmp_path: pathlib.Path):
    tail = RoutingCounterTail(out_path=tmp_path / "routing.jsonl", halt_on_low_noise=False)
    tail.log(step=0, sampled_timestep_id=0, raw_timestep=500, pair_id="p0")
    assert tail.low_count == 1
    summary = tail.summary()
    assert summary["fraction_high_noise"] == 0.0


# ---------- loss_components ----------


def test_loss_logger_records_known_fields(tmp_path: pathlib.Path):
    logger = LossComponentLogger(out_path=tmp_path / "loss.jsonl")
    components = {
        "mse_policy_winner": 0.4,
        "mse_policy_loser": 0.6,
        "mse_reference_winner": 0.5,
        "mse_reference_loser": 0.55,
        "policy_advantage": 0.2,
        "reference_advantage": 0.05,
        "logit": 0.015,
        "implied_winner_prob": 0.504,
    }
    rec = logger.log(step=0, pair_id="abc", t_raw=950, loss=0.69, beta=0.1, components=components)
    for k in KNOWN_COMPONENT_FIELDS:
        assert k in rec
        assert rec[k] is not None
    assert rec["loss"] == 0.69
    assert rec["beta"] == 0.1
    assert rec["t_raw"] == 950


def test_loss_logger_handles_missing_components(tmp_path: pathlib.Path):
    logger = LossComponentLogger(out_path=tmp_path / "loss.jsonl")
    rec = logger.log(step=0, pair_id="abc", t_raw=950, loss=0.5, beta=0.1, components=None)
    for k in KNOWN_COMPONENT_FIELDS:
        assert rec[k] is None


def test_loss_logger_summary_min_max_mean(tmp_path: pathlib.Path):
    logger = LossComponentLogger(out_path=tmp_path / "loss.jsonl")
    for i, lv in enumerate([0.9, 0.6, 0.3, 0.4]):
        logger.log(step=i, pair_id=f"p{i}", t_raw=950, loss=lv, beta=0.1, components={})
    s = logger.summary()
    assert s["steps_logged"] == 4
    assert s["loss_min"] == 0.3
    assert s["loss_max"] == 0.9
    assert s["loss_first"] == 0.9
    assert s["loss_last"] == 0.4
    assert math.isclose(s["loss_mean"], 0.55, rel_tol=1e-6)


def test_loss_logger_marks_nan(tmp_path: pathlib.Path):
    logger = LossComponentLogger(out_path=tmp_path / "loss.jsonl")
    rec = logger.log(step=0, pair_id="p", t_raw=950, loss=float("nan"), beta=0.1)
    assert rec.get("nan_loss") is True
    assert logger.nan_count == 1


# ---------- vram_probe (no-CUDA path) ----------


def test_vram_probe_no_cuda_is_noop(tmp_path: pathlib.Path):
    probe = VramProbe(out_path=tmp_path / "vram.jsonl", enabled=False)
    probe.start_step(0, pair_id="p")
    probe.stage("ref_forward_winner")
    probe.end_step()
    # No record should be written when disabled.
    assert not (tmp_path / "vram.jsonl").exists() or (tmp_path / "vram.jsonl").read_text() == ""
    assert probe.summary()["steps"] == 0


# ---------- aggregator (Watchdog) ----------


def test_watchdog_aggregator_dir_layout(tmp_path: pathlib.Path):
    wd = Watchdog(run_dir=tmp_path, rank=0, halt_on_low_noise=True, enabled=False)
    assert (tmp_path / "watchdog" / "rank0").exists()
    # Drive a synthetic step: routing + loss only, vram disabled.
    wd.log_routing(step=0, sampled_timestep_id=0, raw_timestep=950, pair_id="p0")
    wd.log_loss(step=0, pair_id="p0", t_raw=950, loss=0.7, beta=0.1, components={
        "mse_policy_winner": 0.4,
    })
    summary = wd.flush_summary()
    assert summary["routing"]["high_count"] == 1
    assert summary["loss"]["steps_logged"] == 1
    assert (tmp_path / "watchdog" / "rank0" / "summary.json").exists()


def test_watchdog_halts_when_low_noise(tmp_path: pathlib.Path):
    wd = Watchdog(run_dir=tmp_path, rank=0, halt_on_low_noise=True, enabled=False)
    with pytest.raises(LowNoiseRoutingError):
        wd.log_routing(step=0, sampled_timestep_id=0, raw_timestep=100, pair_id="p")
