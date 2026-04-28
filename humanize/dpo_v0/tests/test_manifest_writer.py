"""Tests for ``manifest_writer``.

Covers the round-1 contract:

* ``assert_recipe_pins`` 3-way assert (positive + four negative paths)
* ``streaming_canonical_hash`` byte-spec lock + idempotency + sidecar
  re-read SHA agreement
* ``write_run_manifest`` atomic write + deterministic key order
* ``halt_judge_axis_missing`` writes audit + raises
* ``RunManifest`` field validation (compute_envelope enum, missing
  required pin keys, negative tensor count)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
PKG_ROOT = HERE.parent  # humanize/dpo_v0/
sys.path.insert(0, str(PKG_ROOT))

from dataprocessing.manifest_writer import (  # noqa: E402
    COMPUTE_ENVELOPES_CANONICAL,
    EXPECTED_AGGREGATION_RULE,
    EXPECTED_DTYPE_POLICY,
    EXPECTED_FPS,
    EXPECTED_FRAME_NUM,
    EXPECTED_SWITCH_DIT_BOUNDARY,
    JUDGE_REQUIRED_AXES,
    KNOWN_GOOD_RECIPE_ID,
    MANIFEST_SCHEMA_VERSION,
    CkptSourcePaths,
    JudgeAxisMissingError,
    JudgeFieldProbe,
    RoutingForwardEntry,
    RunManifest,
    ShardEntry,
    assert_judge_axes_present,
    assert_recipe_pins,
    atomic_write_json,
    halt_judge_axis_missing,
    streaming_canonical_hash,
    write_run_manifest,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_recipes_dir(tmp_path: pathlib.Path, yaml_text: str | None = None) -> pathlib.Path:
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    if yaml_text is None:
        # Build a yaml whose 16-hex hash matches KNOWN_GOOD_RECIPE_ID by
        # picking an arbitrary string and rolling its hash forward; if it
        # doesn't match we recompute a fresh ID and pin to that for the
        # test fixture (the prod value lives in the real repo).
        yaml_text = "fixture: minimal\n"
    yaml_bytes = yaml_text.encode("utf-8")
    fresh = hashlib.sha256(yaml_bytes).hexdigest()[:16]
    (recipes / "wan22_i2v_a14b__round2_v0.yaml").write_bytes(yaml_bytes)
    (recipes / "recipe_id").write_text(fresh, encoding="ascii")
    return recipes


def _basic_manifest_kwargs(sidecar_path: pathlib.Path) -> dict:
    return dict(
        timestamp="2026-04-27T22:00:00+00:00",
        commit_id="0123456789abcdef0123456789abcdef01234567",
        machine_internal_ip_tail=".196",
        compute_envelope="dpo_multi_gpu_ddp",
        ckpt_source_paths=CkptSourcePaths(
            high_noise_base="/UPSTREAM/Wan2.2-I2V-A14B/high_noise_model",
            low_noise_frozen="/UPSTREAM/Wan2.2-I2V-A14B/low_noise_model",
        ),
        shard_manifest=[
            ShardEntry(file="shard_001.safetensors", sha256="a" * 64, param_count=10, dtype="F32"),
        ],
        merged_state_sha256="b" * 64,
        per_key_sidecar_path=str(sidecar_path),
        per_key_sidecar_sha256="c" * 64,
        tensor_count=1,
        recipe_pins={
            "recipe_id": KNOWN_GOOD_RECIPE_ID,
            "switch_DiT_boundary": EXPECTED_SWITCH_DIT_BOUNDARY,
            "fps": EXPECTED_FPS,
            "frame_num": EXPECTED_FRAME_NUM,
            "aggregation_rule": EXPECTED_AGGREGATION_RULE,
            "dtype_policy": EXPECTED_DTYPE_POLICY,
        },
        routing_counter_log=[
            RoutingForwardEntry(sampled_timestep_id=0, raw_timestep=950, detected_expert="high_noise"),
        ],
        judge_field_probe=JudgeFieldProbe(
            axis_to_field={a: a for a in JUDGE_REQUIRED_AXES},
            raw_probe_payload={"SA": 0.5, "PTV": 0.6, "persistence": 0.7},
        ),
        generation_config={"seed": 0, "inference_steps": 50},
    )


# ---------------------------------------------------------------------------
# assert_recipe_pins
# ---------------------------------------------------------------------------


def test_assert_recipe_pins_happy_path(tmp_path):
    recipes = _make_recipes_dir(tmp_path)
    fresh = hashlib.sha256((recipes / "wan22_i2v_a14b__round2_v0.yaml").read_bytes()).hexdigest()[:16]
    pins = assert_recipe_pins(recipes, expected_recipe_id=fresh)
    assert pins["recipe_id"] == fresh
    assert pins["switch_DiT_boundary"] == EXPECTED_SWITCH_DIT_BOUNDARY
    assert pins["fps"] == EXPECTED_FPS
    assert pins["frame_num"] == EXPECTED_FRAME_NUM


def test_assert_recipe_pins_pin_vs_yaml_mismatch_halts(tmp_path):
    recipes = _make_recipes_dir(tmp_path)
    # Tamper with the on-disk pin so it disagrees with the recompute.
    (recipes / "recipe_id").write_text("deadbeefdeadbeef", encoding="ascii")
    with pytest.raises(AssertionError, match="recipe pin drift"):
        assert_recipe_pins(recipes, expected_recipe_id="deadbeefdeadbeef")


def test_assert_recipe_pins_constant_mismatch_halts(tmp_path):
    recipes = _make_recipes_dir(tmp_path)
    # Pin and yaml agree, but the constant is wrong.
    with pytest.raises(AssertionError, match="recipe pin drift"):
        assert_recipe_pins(recipes, expected_recipe_id="deadbeefdeadbeef")


def test_assert_recipe_pins_yaml_drifts_pin_stale_halts(tmp_path):
    recipes = _make_recipes_dir(tmp_path)
    fresh = hashlib.sha256((recipes / "wan22_i2v_a14b__round2_v0.yaml").read_bytes()).hexdigest()[:16]
    # Mutate the YAML so the recompute drifts even though pin file is stale.
    (recipes / "wan22_i2v_a14b__round2_v0.yaml").write_bytes(b"fixture: tampered\n")
    with pytest.raises(AssertionError, match="recipe pin drift"):
        assert_recipe_pins(recipes, expected_recipe_id=fresh)


def test_assert_recipe_pins_missing_pin_file_raises(tmp_path):
    recipes = _make_recipes_dir(tmp_path)
    (recipes / "recipe_id").unlink()
    with pytest.raises(FileNotFoundError):
        assert_recipe_pins(recipes, expected_recipe_id=KNOWN_GOOD_RECIPE_ID)


# ---------------------------------------------------------------------------
# streaming_canonical_hash
# ---------------------------------------------------------------------------


def _two_tiny_tensors() -> list[tuple[str, torch.Tensor]]:
    return [
        ("alpha", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)),
        ("beta", torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)),
    ]


def test_streaming_hash_byte_spec_lock(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    merged_a, sidecar_sha_a, count_a = streaming_canonical_hash(_two_tiny_tensors(), sidecar)
    assert count_a == 2

    # Second pass over identical inputs: byte-equal sidecar + identical hashes.
    sidecar2 = tmp_path / "sidecar2.jsonl"
    merged_b, sidecar_sha_b, count_b = streaming_canonical_hash(_two_tiny_tensors(), sidecar2)
    assert merged_a == merged_b
    assert sidecar_sha_a == sidecar_sha_b
    assert count_b == 2
    assert sidecar.read_bytes() == sidecar2.read_bytes()


def test_streaming_hash_sidecar_jsonl_alphabetical(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    streaming_canonical_hash(_two_tiny_tensors(), sidecar)
    lines = sidecar.read_text().strip().split("\n")
    assert len(lines) == 2
    keys = [json.loads(line)["key"] for line in lines]
    assert keys == ["alpha", "beta"]


def test_streaming_hash_per_key_uses_fresh_hasher(tmp_path):
    """Verify per-key SHAs are independent (not running totals)."""
    sidecar = tmp_path / "sidecar.jsonl"
    streaming_canonical_hash(_two_tiny_tensors(), sidecar)
    rows = [json.loads(line) for line in sidecar.read_text().strip().split("\n")]
    sha_a, sha_b = rows[0]["sha256"], rows[1]["sha256"]

    # Recompute per-key alpha sha from scratch and compare.
    h = hashlib.sha256()
    h.update(b"alpha|(3,)|torch.float32|")
    h.update(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy().tobytes())
    assert sha_a == h.hexdigest()
    assert sha_a != sha_b


# ---------------------------------------------------------------------------
# write_run_manifest + RunManifest validation
# ---------------------------------------------------------------------------


def test_write_run_manifest_round_trip(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text("")  # placeholder
    manifest = RunManifest(**_basic_manifest_kwargs(sidecar))
    out_dir = tmp_path / "run"
    path = write_run_manifest(out_dir, manifest)
    assert path == out_dir / "manifest.json"
    data = json.loads(path.read_text())
    assert data["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert data["compute_envelope"] == "dpo_multi_gpu_ddp"
    assert data["recipe_pins"]["recipe_id"] == KNOWN_GOOD_RECIPE_ID
    assert data["routing_counter_log"][0]["detected_expert"] == "high_noise"


def test_write_run_manifest_atomic_no_partial_replacement(tmp_path):
    """The manifest path either does not exist or is fully written."""
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text("")
    manifest = RunManifest(**_basic_manifest_kwargs(sidecar))
    out_dir = tmp_path / "run"
    write_run_manifest(out_dir, manifest)
    final = out_dir / "manifest.json"
    assert final.exists()
    # No leftover temp file.
    assert not (out_dir / "manifest.json.tmp").exists()


def test_run_manifest_unknown_compute_envelope_raises(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text("")
    kwargs = _basic_manifest_kwargs(sidecar)
    kwargs["compute_envelope"] = "wishful_thinking"
    with pytest.raises(ValueError, match="unknown compute_envelope"):
        RunManifest(**kwargs)


def test_run_manifest_known_envelopes_round_trip(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text("")
    for env in COMPUTE_ENVELOPES_CANONICAL:
        kwargs = _basic_manifest_kwargs(sidecar)
        kwargs["compute_envelope"] = env
        m = RunManifest(**kwargs)
        assert m.compute_envelope == env


def test_run_manifest_missing_recipe_pin_key_raises(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text("")
    kwargs = _basic_manifest_kwargs(sidecar)
    kwargs["recipe_pins"] = {"recipe_id": KNOWN_GOOD_RECIPE_ID}  # drop other keys
    with pytest.raises(ValueError, match="recipe_pins missing required key"):
        RunManifest(**kwargs)


def test_run_manifest_negative_tensor_count_raises(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text("")
    kwargs = _basic_manifest_kwargs(sidecar)
    kwargs["tensor_count"] = -1
    with pytest.raises(ValueError, match="tensor_count must be non-negative"):
        RunManifest(**kwargs)


# ---------------------------------------------------------------------------
# halt_judge_axis_missing + assert_judge_axes_present
# ---------------------------------------------------------------------------


def test_halt_judge_axis_missing_writes_audit_and_raises(tmp_path):
    out_dir = tmp_path / "run"
    with pytest.raises(JudgeAxisMissingError, match="missing required axes"):
        halt_judge_axis_missing(["PTV"], out_dir, raw_probe_payload={"SA": 0.5})
    audit = out_dir / "judge_axis_missing.json"
    assert audit.exists()
    payload = json.loads(audit.read_text())
    assert payload["missing_axes"] == ["PTV"]
    assert payload["halt_reason"] == "judge-axis-missing"
    assert "schema_version" in payload


def test_assert_judge_axes_present_happy_path(tmp_path):
    out_dir = tmp_path / "run"
    probe = {"SA": 1.0, "PTV": 1.0, "persistence": 1.0}
    result = assert_judge_axes_present(probe, out_dir)
    assert result.axis_to_field == {"SA": "SA", "PTV": "PTV", "persistence": "persistence"}
    assert not (out_dir / "judge_axis_missing.json").exists()


def test_assert_judge_axes_present_with_field_remap(tmp_path):
    out_dir = tmp_path / "run"
    probe = {"semantic_alignment": 1.0, "physics_temporal_value": 1.0, "persistence": 1.0}
    remap = {"SA": "semantic_alignment", "PTV": "physics_temporal_value", "persistence": "persistence"}
    result = assert_judge_axes_present(probe, out_dir, axis_to_field=remap)
    assert result.axis_to_field == remap


def test_assert_judge_axes_present_missing_axis_halts(tmp_path):
    out_dir = tmp_path / "run"
    probe = {"SA": 1.0, "PTV": 1.0}  # persistence missing
    with pytest.raises(JudgeAxisMissingError):
        assert_judge_axes_present(probe, out_dir)
    assert (out_dir / "judge_axis_missing.json").exists()


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


def test_atomic_write_json_deterministic_key_order(tmp_path):
    p = tmp_path / "out.json"
    obj = {"b": 1, "a": 2}
    atomic_write_json(p, obj)
    text = p.read_text()
    # sort_keys=True forces "a" to appear before "b".
    assert text.index('"a"') < text.index('"b"')
