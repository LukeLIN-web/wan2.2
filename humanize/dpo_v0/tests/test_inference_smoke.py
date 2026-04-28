"""Tests for inference_smoke (M5 / AC-7.0 dual-sample smoke).

Covers the round-2 contracts that downstream consumers (M6 heldout regen
orchestrator's subprocess + python_api adapters; M8 PhyJudge field probe)
depend on:
  - CLI shape lock per `5e8993eb` (mode / gen_config / low_noise_ckpt / etc.)
  - Mode-conditioned argument validation
  - byte-identical `generation_config` round-trip via `--gen-config-json`
  - sharded ckpt SHA stability + sensitivity to bytes / order / filename
  - public `run_one_sample` keyword-arg signature for python_api adapter
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # humanize/dpo_v0/
from eval.inference_smoke import (  # noqa: E402
    COMPUTE_ENVELOPES_CANONICAL,
    EXPECTED_RECIPE_ID,
    FPS,
    NUM_FRAMES,
    SWITCH_DIT_BOUNDARY,
    UpstreamPaths,
    assert_recipe_pin,
    build_generation_config,
    file_sha256,
    parse_args,
    run_one_sample,
    serialize_generation_config,
    sharded_ckpt_sha,
)


# --------------------------------------------------------------------------
# 1. Recipe pin assertion (3-way: constant + on-disk + recompute YAML)
# --------------------------------------------------------------------------


def test_recipe_pin_default_passes():
    rid = assert_recipe_pin()
    assert rid == EXPECTED_RECIPE_ID


def test_recipe_pin_drift_raises(tmp_path: pathlib.Path):
    (tmp_path / "wan22_i2v_a14b__round2_v0.yaml").write_bytes(b"# bogus yaml\n")
    (tmp_path / "recipe_id").write_text(EXPECTED_RECIPE_ID, encoding="ascii")
    with pytest.raises(RuntimeError, match="recipe_id pin drift"):
        assert_recipe_pin(recipes_dir=tmp_path)


# --------------------------------------------------------------------------
# 2. sharded_ckpt_sha: stable across walk order, sensitive to content
# --------------------------------------------------------------------------


def _write_shard(p: pathlib.Path, payload: bytes) -> pathlib.Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


def test_sharded_ckpt_sha_stable_across_input_order(tmp_path: pathlib.Path):
    a = _write_shard(tmp_path / "shard-a.safetensors", b"AAA")
    b = _write_shard(tmp_path / "shard-b.safetensors", b"BBB")
    c = _write_shard(tmp_path / "shard-c.safetensors", b"CCC")
    sha_alpha = sharded_ckpt_sha([a, b, c])
    sha_reverse = sharded_ckpt_sha([c, b, a])
    sha_shuffled = sharded_ckpt_sha([b, a, c])
    assert sha_alpha == sha_reverse == sha_shuffled


def test_sharded_ckpt_sha_changes_on_byte_swap(tmp_path: pathlib.Path):
    a = _write_shard(tmp_path / "shard-a.safetensors", b"AAA")
    b = _write_shard(tmp_path / "shard-b.safetensors", b"BBB")
    sha_before = sharded_ckpt_sha([a, b])
    a.write_bytes(b"DDD")  # mutate one shard's content
    sha_after = sharded_ckpt_sha([a, b])
    assert sha_before != sha_after


def test_sharded_ckpt_sha_changes_on_filename_rename(tmp_path: pathlib.Path):
    p1 = _write_shard(tmp_path / "shard-a.safetensors", b"AAA")
    sha1 = sharded_ckpt_sha([p1])
    p2 = p1.rename(tmp_path / "shard-renamed.safetensors")
    sha2 = sharded_ckpt_sha([p2])
    assert sha1 != sha2


def test_sharded_ckpt_sha_changes_on_shard_added(tmp_path: pathlib.Path):
    a = _write_shard(tmp_path / "shard-a.safetensors", b"AAA")
    b = _write_shard(tmp_path / "shard-b.safetensors", b"BBB")
    sha_two = sharded_ckpt_sha([a, b])
    c = _write_shard(tmp_path / "shard-c.safetensors", b"CCC")
    sha_three = sharded_ckpt_sha([a, b, c])
    assert sha_two != sha_three


# --------------------------------------------------------------------------
# 3. CLI argparse: mode + gen_config_json + envelope + recipe_yaml
# --------------------------------------------------------------------------


def _baseline_args(tmp_path: pathlib.Path, **overrides) -> list[str]:
    """Build a CLI arg list with placeholder paths sufficient for parse_args.

    parse_args does not check filesystem presence (that's main()'s job), so
    we can pass any string and exercise only the argparse-level validation.
    """
    base = {
        "--upstream": str(tmp_path / "upstream"),
        "--prompt": "hello world",
        "--cond-image": str(tmp_path / "cond.jpg"),
        "--out-dir": str(tmp_path / "out"),
    }
    base.update(overrides)
    argv: list[str] = []
    for k, v in base.items():
        if v is None:
            continue
        argv.extend([k, str(v)])
    return argv


def test_argparse_default_mode_is_both(tmp_path: pathlib.Path):
    argv = _baseline_args(tmp_path, **{"--lora-adapter": str(tmp_path / "lora.safetensors")})
    args = parse_args(argv)
    assert args.mode == "both"


def test_argparse_mode_baseline_rejects_lora_adapter(tmp_path: pathlib.Path):
    argv = _baseline_args(
        tmp_path,
        **{"--mode": "baseline", "--lora-adapter": str(tmp_path / "lora.safetensors")},
    )
    with pytest.raises(SystemExit, match="must not be passed --lora-adapter"):
        parse_args(argv)


def test_argparse_mode_baseline_no_lora_required(tmp_path: pathlib.Path):
    argv = _baseline_args(tmp_path, **{"--mode": "baseline"})
    args = parse_args(argv)
    assert args.mode == "baseline" and args.lora_adapter is None


def test_argparse_mode_trained_requires_lora_adapter(tmp_path: pathlib.Path):
    argv = _baseline_args(tmp_path, **{"--mode": "trained"})
    with pytest.raises(SystemExit, match="requires --lora-adapter"):
        parse_args(argv)


def test_argparse_mode_both_requires_lora_adapter(tmp_path: pathlib.Path):
    # Default mode is "both"; no --lora-adapter flag at all.
    argv = _baseline_args(tmp_path)
    with pytest.raises(SystemExit, match="requires --lora-adapter"):
        parse_args(argv)


def test_argparse_envelope_enum_includes_all_canonical(tmp_path: pathlib.Path):
    # Each canonical envelope value must be argparse-accepted.
    for env in COMPUTE_ENVELOPES_CANONICAL:
        argv = _baseline_args(
            tmp_path,
            **{
                "--mode": "baseline",
                "--compute-envelope": env,
            },
        )
        args = parse_args(argv)
        assert args.compute_envelope == env


def test_argparse_unknown_envelope_rejected(tmp_path: pathlib.Path):
    argv = _baseline_args(
        tmp_path,
        **{"--mode": "baseline", "--compute-envelope": "not_a_real_envelope"},
    )
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_argparse_switch_dit_boundary_locked(tmp_path: pathlib.Path):
    argv = _baseline_args(
        tmp_path,
        **{
            "--mode": "baseline",
            "--switch-DiT-boundary": "0.85",
        },
    )
    with pytest.raises(SystemExit, match="must be 0.9"):
        parse_args(argv)


def test_argparse_num_frames_locked(tmp_path: pathlib.Path):
    argv = _baseline_args(
        tmp_path, **{"--mode": "baseline", "--num-frames": "65"}
    )
    with pytest.raises(SystemExit, match=f"must be {NUM_FRAMES}"):
        parse_args(argv)


def test_argparse_fps_locked(tmp_path: pathlib.Path):
    argv = _baseline_args(
        tmp_path, **{"--mode": "baseline", "--fps": "24"}
    )
    with pytest.raises(SystemExit, match=f"must be {FPS}"):
        parse_args(argv)


# --------------------------------------------------------------------------
# 4. byte-identical generation_config: serializer + dual-call equality
# --------------------------------------------------------------------------


def _make_namespace(tmp_path: pathlib.Path) -> "argparse.Namespace":  # noqa: F821
    """Build a Namespace with all fields needed by build_generation_config."""
    import argparse as _ap
    return _ap.Namespace(
        seed=0,
        num_inference_steps=50,
        cfg_scale=5.0,
        negative_prompt="negative",
        height=480,
        width=832,
        num_frames=NUM_FRAMES,
        fps=FPS,
        torch_dtype="bfloat16",
        switch_DiT_boundary=SWITCH_DIT_BOUNDARY,
    )


def test_serialize_generation_config_byte_stable(tmp_path: pathlib.Path):
    ns = _make_namespace(tmp_path)
    cfg = build_generation_config(ns)
    b1 = serialize_generation_config(cfg)
    b2 = serialize_generation_config(cfg)
    assert b1 == b2  # same dict -> same bytes
    assert b1.endswith(b"\n")
    # round-trip through json.loads should give back the same dict
    assert json.loads(b1) == cfg


def test_serialize_generation_config_diff_seed_changes_bytes(tmp_path: pathlib.Path):
    ns_a = _make_namespace(tmp_path)
    ns_b = _make_namespace(tmp_path)
    ns_b.seed = 42  # one-field flip
    a = serialize_generation_config(build_generation_config(ns_a))
    b = serialize_generation_config(build_generation_config(ns_b))
    assert a != b
    # And the digests should differ -- this is what the M6 orchestrator's
    # `gen_config_sha256` hard contract relies on for drift detection.
    assert hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------
# 5. run_one_sample keyword-arg validation (the python_api adapter contract)
# --------------------------------------------------------------------------


def test_run_one_sample_rejects_baseline_with_lora(tmp_path: pathlib.Path):
    """python_api adapter: passing a lora path with mode='baseline' must
    raise rather than silently ignoring the LoRA (which would taint the
    baseline run for AC-7.3 audit purposes)."""
    upstream = UpstreamPaths(upstream_root=tmp_path)
    with pytest.raises(ValueError, match="must not be passed a lora_adapter_path"):
        run_one_sample(
            mode="baseline",
            out_dir=tmp_path / "out",
            upstream=upstream,
            torch_dtype=__import__("torch").bfloat16,
            device="cpu",
            prompt="x",
            cond_image_path=tmp_path / "cond.jpg",
            generation_config={},
            generation_config_bytes=b"{}",
            lora_adapter_path=tmp_path / "lora.safetensors",
            compute_envelope="single_gpu",
            recipe_id=EXPECTED_RECIPE_ID,
        )


def test_run_one_sample_rejects_trained_without_lora(tmp_path: pathlib.Path):
    upstream = UpstreamPaths(upstream_root=tmp_path)
    with pytest.raises(ValueError, match="requires lora_adapter_path"):
        run_one_sample(
            mode="trained",
            out_dir=tmp_path / "out",
            upstream=upstream,
            torch_dtype=__import__("torch").bfloat16,
            device="cpu",
            prompt="x",
            cond_image_path=tmp_path / "cond.jpg",
            generation_config={},
            generation_config_bytes=b"{}",
            lora_adapter_path=None,
            compute_envelope="single_gpu",
            recipe_id=EXPECTED_RECIPE_ID,
        )


def test_run_one_sample_rejects_invalid_mode(tmp_path: pathlib.Path):
    upstream = UpstreamPaths(upstream_root=tmp_path)
    with pytest.raises(ValueError, match="mode must be 'baseline' or 'trained'"):
        run_one_sample(
            mode="both",  # type: ignore[arg-type]
            out_dir=tmp_path / "out",
            upstream=upstream,
            torch_dtype=__import__("torch").bfloat16,
            device="cpu",
            prompt="x",
            cond_image_path=tmp_path / "cond.jpg",
            generation_config={},
            generation_config_bytes=b"{}",
            lora_adapter_path=None,
            compute_envelope="single_gpu",
            recipe_id=EXPECTED_RECIPE_ID,
        )
