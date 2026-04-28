"""Smoke tests for the canonical sharded loader.

The tests use small in-memory state dicts (no real 75 GB shards) to
exercise the streaming hash field order, alphabetical walk, and
recipe-pin assertion. Hardcoded expected SHAs lock the AC-3.4 byte
spectrum (PyTorch ``str(dtype)`` form, ``repr(tuple(shape))`` form,
field separator) so a future PyTorch version that changes any of
those reprs fails this test rather than silently drifting.

Negative tests exercise the AC-1 abort paths: missing shard file,
shard mis-mapping in ``safetensors_index.json``, divergent / malformed
``safetensors_index.json``, and an unrecognized expert directory name.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile

import pytest
import safetensors.torch
import torch

from canonical_loader import (
    FIELD_SEPARATOR,
    KNOWN_GOOD_RECIPE_ID,
    SCHEMA_VERSION,
    StreamingHasher,
    canonical_field_bytes,
    load_expert,
    per_key_sidecar,
    streaming_merged_sha256,
    _enumerate_shards,
    _read_recipe_pin,
    _resolve_expert_tag,
)


# ---------- Hardcoded fixture SHAs (lock the AC-3.4 byte spectrum) ----------

EXPECTED_PER_KEY_SHA = {
    "alpha": "6f96dbdbbcae404d026061319071ca442df6f2c44a7575d9628c7d2677248b87",
    "beta": "13c478754549f532e615838459ae27cc14e89950db6df25efc3f08cbce8b5f52",
    "gamma": "0330d2082b1a99ec148d24c671a533812d3ef5b09b0b7359e9b334b92c9f97cf",
}
EXPECTED_MERGED_SHA = "14c5c6a235105f3fb2e561c0788d88bb66fc7aff5f340a9368b66d2e64c155ad"


def _mock_state() -> dict[str, torch.Tensor]:
    return {
        "alpha": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "beta": torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32),
        "gamma": torch.tensor(42, dtype=torch.int64),
    }


def _reference_per_key_sha(key: str, tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    body = contiguous.numpy().tobytes()
    key_b, shape_b, dtype_b = canonical_field_bytes(key, tuple(contiguous.shape), str(contiguous.dtype))
    full = (
        key_b
        + FIELD_SEPARATOR
        + shape_b
        + FIELD_SEPARATOR
        + dtype_b
        + FIELD_SEPARATOR
        + body
    )
    return hashlib.sha256(full).hexdigest()


def _reference_merged_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        contiguous = state[key].detach().cpu().contiguous()
        body = contiguous.numpy().tobytes()
        key_b, shape_b, dtype_b = canonical_field_bytes(key, tuple(contiguous.shape), str(contiguous.dtype))
        h.update(key_b + FIELD_SEPARATOR + shape_b + FIELD_SEPARATOR + dtype_b + FIELD_SEPARATOR + body)
    return h.hexdigest()


# ---------- Streaming hash ----------


def test_streaming_matches_reference():
    state = _mock_state()
    streaming, count = streaming_merged_sha256(state)
    reference = _reference_merged_sha(state)
    assert streaming == reference, (streaming, reference)
    assert count == 3


def test_merged_sha_matches_hardcoded_fixture():
    """Locks AC-3.4 byte spectrum: PyTorch dtype repr, shape repr, separator."""
    state = _mock_state()
    streaming, _ = streaming_merged_sha256(state)
    assert streaming == EXPECTED_MERGED_SHA, (
        f"merged sha drift; if this fails, PyTorch likely changed "
        f"str(torch.dtype) or repr(tuple(shape)). expected={EXPECTED_MERGED_SHA} got={streaming}"
    )


def test_per_key_shas_match_hardcoded_fixtures():
    state = _mock_state()
    for key, expected in EXPECTED_PER_KEY_SHA.items():
        got = _reference_per_key_sha(key, state[key])
        assert got == expected, f"per-key sha drift for {key}: expected={expected} got={got}"


def test_alphabetical_walk_independent_of_insertion_order():
    s_ordered = _mock_state()
    s_reversed = {k: s_ordered[k] for k in reversed(list(s_ordered))}
    sha_a, _ = streaming_merged_sha256(s_ordered)
    sha_b, _ = streaming_merged_sha256(s_reversed)
    assert sha_a == sha_b


def test_streaming_hasher_per_key_returns_match_reference():
    state = _mock_state()
    h = StreamingHasher()
    per_key = {key: h.update(key, state[key]) for key in sorted(state)}
    for key, sha in per_key.items():
        assert sha == _reference_per_key_sha(key, state[key]), key


# ---------- Sidecar JSONL ----------


def test_per_key_sidecar_lines_alphabetical_and_hash_matches():
    state = _mock_state()
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "sidecar.jsonl"
        sidecar_sha, count = per_key_sidecar(state, out)
        assert count == 3
        lines = out.read_bytes().splitlines()
        assert len(lines) == 3
        keys = [json.loads(line)["key"] for line in lines]
        assert keys == sorted(state)
        for line in lines:
            entry = json.loads(line)
            tensor = state[entry["key"]]
            assert entry["sha256"] == _reference_per_key_sha(entry["key"], tensor)
        rebuilt = bytearray()
        for line in lines:
            rebuilt += line + b"\n"
        assert hashlib.sha256(bytes(rebuilt)).hexdigest() == sidecar_sha


def test_per_key_sidecar_idempotent():
    state = _mock_state()
    with tempfile.TemporaryDirectory() as tmp:
        out_a = pathlib.Path(tmp) / "a.jsonl"
        out_b = pathlib.Path(tmp) / "b.jsonl"
        sha_a, _ = per_key_sidecar(state, out_a)
        sha_b, _ = per_key_sidecar(state, out_b)
        assert sha_a == sha_b
        assert out_a.read_bytes() == out_b.read_bytes()


# ---------- Recipe pin (positive + negative) ----------


def _write_recipe_dir(tmp: pathlib.Path, yaml_bytes: bytes, id_text: str) -> pathlib.Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "wan22_i2v_a14b__round2_v0.yaml").write_bytes(yaml_bytes)
    (tmp / "recipe_id").write_text(id_text, encoding="ascii")
    return tmp


def test_recipe_pin_round_trip_no_expected():
    sample_yaml = b"foo: bar\n"
    expected = hashlib.sha256(sample_yaml).hexdigest()[:16]
    with tempfile.TemporaryDirectory() as tmp:
        recipes_dir = _write_recipe_dir(pathlib.Path(tmp), sample_yaml, expected + "\n")
        assert _read_recipe_pin(recipes_dir) == expected


def test_recipe_pin_known_good_drift_raises():
    """When the on-disk id mismatches a passed-in known-good, raise."""
    sample_yaml = b"foo: bar\n"
    expected = hashlib.sha256(sample_yaml).hexdigest()[:16]
    with tempfile.TemporaryDirectory() as tmp:
        recipes_dir = _write_recipe_dir(pathlib.Path(tmp), sample_yaml, expected + "\n")
        # on-disk yaml/id are internally consistent, but the caller passes a
        # different known-good (e.g., the loader's KNOWN_GOOD_RECIPE_ID).
        with pytest.raises(ValueError, match="recipe id drift"):
            _read_recipe_pin(recipes_dir, expected="0" * 16)


def test_recipe_pin_internal_mismatch_raises():
    sample_yaml = b"foo: bar\n"
    bogus_id = "0" * 16
    with tempfile.TemporaryDirectory() as tmp:
        recipes_dir = _write_recipe_dir(pathlib.Path(tmp), sample_yaml, bogus_id + "\n")
        with pytest.raises(ValueError, match="recipe id mismatch"):
            _read_recipe_pin(recipes_dir)


def test_recipe_pin_bad_chars_raises():
    sample_yaml = b"foo: bar\n"
    with tempfile.TemporaryDirectory() as tmp:
        recipes_dir = _write_recipe_dir(pathlib.Path(tmp), sample_yaml, "not_hex_at_all\n")
        with pytest.raises(ValueError, match="not 16 hex chars"):
            _read_recipe_pin(recipes_dir)


def test_recipe_pin_missing_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            _read_recipe_pin(pathlib.Path(tmp))


# ---------- Expert tag resolution (negative) ----------


def test_unrecognized_expert_dir_name_raises(tmp_path):
    bogus_dir = tmp_path / "wrong_name"
    bogus_dir.mkdir()
    with pytest.raises(ValueError, match="unrecognized expert directory name"):
        _resolve_expert_tag(bogus_dir)


def test_high_noise_dir_resolves():
    assert _resolve_expert_tag(pathlib.Path("/tmp/high_noise_model")) == "high_noise"


def test_low_noise_dir_resolves():
    assert _resolve_expert_tag(pathlib.Path("/tmp/low_noise_model")) == "low_noise"


# ---------- Shard enumeration negative tests (AC-1 abort paths) ----------


def _make_minimal_expert(tmp: pathlib.Path, name: str = "high_noise_model") -> pathlib.Path:
    """Write 2 minimal safetensor shards + a valid index for negative-test setup."""
    expert_dir = tmp / name
    expert_dir.mkdir(parents=True)
    state_a = {"layer0.weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    state_b = {"layer1.weight": torch.tensor([3.0, 4.0], dtype=torch.float32)}
    safetensors.torch.save_file(state_a, str(expert_dir / "diffusion_pytorch_model-00001-of-00002.safetensors"))
    safetensors.torch.save_file(state_b, str(expert_dir / "diffusion_pytorch_model-00002-of-00002.safetensors"))
    index = {
        "metadata": {"total_size": 8 + 8},
        "weight_map": {
            "layer0.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
            "layer1.weight": "diffusion_pytorch_model-00002-of-00002.safetensors",
        },
    }
    (expert_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    return expert_dir


def test_negative_missing_shard_file(tmp_path):
    """AC-1: a missing shard file aborts before identity gate."""
    expert_dir = _make_minimal_expert(tmp_path)
    (expert_dir / "diffusion_pytorch_model-00002-of-00002.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="shard missing"):
        _enumerate_shards(expert_dir)


def test_negative_missing_index_json(tmp_path):
    """AC-1: a missing safetensors_index.json aborts."""
    expert_dir = _make_minimal_expert(tmp_path)
    (expert_dir / "diffusion_pytorch_model.safetensors.index.json").unlink()
    with pytest.raises(FileNotFoundError, match="safetensors index missing"):
        _enumerate_shards(expert_dir)


def test_negative_index_points_to_wrong_shard(tmp_path):
    """AC-1: a key mapped to a shard that does not actually contain it aborts."""
    expert_dir = _make_minimal_expert(tmp_path)
    # Recipes dir to satisfy the recipe pin gate before _enumerate_shards runs.
    recipes_dir = tmp_path / "recipes"
    sample_yaml = b"foo: bar\n"
    _write_recipe_dir(recipes_dir, sample_yaml, hashlib.sha256(sample_yaml).hexdigest()[:16] + "\n")
    # Swap weight_map so layer0.weight points at the shard that doesn't have it.
    index_path = expert_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["layer0.weight"] = "diffusion_pytorch_model-00002-of-00002.safetensors"
    index["weight_map"]["layer1.weight"] = "diffusion_pytorch_model-00001-of-00002.safetensors"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    manifest_out = tmp_path / "manifest.json"
    sidecar_out = tmp_path / "sidecar.jsonl"
    # safetensors raises a generic exception when a key is missing from a shard;
    # we accept any non-zero exit (the abort is what matters, not the exact type).
    with pytest.raises(Exception):
        load_expert(
            expert_dir, manifest_out, sidecar_out, recipes_dir,
            expected_recipe_id=None,
        )


def test_negative_malformed_index_json(tmp_path):
    """AC-1: a divergent (malformed) safetensors_index.json aborts."""
    expert_dir = _make_minimal_expert(tmp_path)
    (expert_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )
    with pytest.raises(json.JSONDecodeError):
        _enumerate_shards(expert_dir)


# ---------- Schema invariants ----------


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


def test_known_good_recipe_id_format():
    assert len(KNOWN_GOOD_RECIPE_ID) == 16
    assert all(c in "0123456789abcdef" for c in KNOWN_GOOD_RECIPE_ID)
