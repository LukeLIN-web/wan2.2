"""Heldout regen orchestration scaffold tests.

Pure-Python tests for the orchestration logic — no GPU, no real
inference. Uses the dry-run inference adapter and synthetic heldout +
T2 manifest fixtures that mirror the real schema.

Run from videodpoWan repo root::

    cd /shared/user60/worldmodel/rlvideo/videodpoWan
    python -m pytest humanize/dpo_v0/tests/test_heldout_regen.py -v
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
PKG_ROOT = HERE.parent          # humanize/dpo_v0/
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(PKG_ROOT.parent.parent))  # videodpoWan

import heldout_regen  # noqa: E402


# ---------- fixtures ----------


def _build_heldout_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Synthesize a T0_T3_ROOT with the right shape (42 prompts / 245 groups / 579 pairs)."""
    root = tmp_path / "t0_t3"
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "t2").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    image_manifest: dict[str, dict] = {}
    pair_idx = 0
    group_idx = 0
    for p in range(heldout_regen.EXPECTED_HELDOUT_PROMPTS):
        prompt = f"prompt {p}: a thing happens"
        # Spread 245 groups across 42 prompts; 245/42 ≈ 5.83.
        n_groups = 6 if p < (heldout_regen.EXPECTED_HELDOUT_GROUPS - 5 * heldout_regen.EXPECTED_HELDOUT_PROMPTS) else 5
        # Adjust so total is exactly 245.
        n_groups = 6 if p < 5 else 5  # 5*6 + 37*5 = 30 + 185 = 215; need 245
        n_groups = 6 if p < 35 else 5  # 35*6 + 7*5 = 210 + 35 = 245 ✓
        for g in range(n_groups):
            group_id = f"g{group_idx:04d}"
            group_idx += 1
            image_manifest[group_id] = {
                "image_path": f"/fake/{group_id}.png",
                "scene_filename": f"scene_{group_id}.mp4",
                "status": "ok",
            }
            # ~2.36 pairs per group on avg, target 579 total
            for _ in range(2 if pair_idx + 1 < heldout_regen.EXPECTED_HELDOUT_PAIRS else 1):
                if pair_idx >= heldout_regen.EXPECTED_HELDOUT_PAIRS:
                    break
                records.append(
                    {
                        "pair_id": f"{group_id}__pair{pair_idx:04d}",
                        "group_id": group_id,
                        "prompt": prompt,
                        "split": "heldout",
                    }
                )
                pair_idx += 1
    # Fill the remainder one-pair-per-group cycling through groups.
    group_keys = list(image_manifest.keys())
    while pair_idx < heldout_regen.EXPECTED_HELDOUT_PAIRS:
        gid = group_keys[pair_idx % len(group_keys)]
        # Find the prompt that owns this group.
        prompt = next(r["prompt"] for r in records if r["group_id"] == gid)
        records.append(
            {
                "pair_id": f"{gid}__filler{pair_idx:04d}",
                "group_id": gid,
                "prompt": prompt,
                "split": "heldout",
            }
        )
        pair_idx += 1

    assert len(records) == heldout_regen.EXPECTED_HELDOUT_PAIRS
    assert len({r["group_id"] for r in records}) == heldout_regen.EXPECTED_HELDOUT_GROUPS
    assert len({r["prompt"] for r in records}) == heldout_regen.EXPECTED_HELDOUT_PROMPTS

    (root / "splits" / "heldout.json").write_text(json.dumps(records))
    (root / "t2" / "image_manifest.json").write_text(json.dumps(image_manifest))
    return root


# ---------- canonical-selection tests ----------


def test_load_heldout_enforces_schema(tmp_path: pathlib.Path):
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    assert len(records) == heldout_regen.EXPECTED_HELDOUT_PAIRS


def test_load_heldout_rejects_wrong_count(tmp_path: pathlib.Path):
    root = tmp_path / "bad"
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "heldout.json").write_text(json.dumps([{"pair_id": "x", "prompt": "y", "group_id": "g"}]))
    with pytest.raises(ValueError, match="pair count mismatch"):
        heldout_regen.load_heldout_records(root)


def test_select_canonical_groups_returns_42(tmp_path: pathlib.Path):
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    selections = heldout_regen.select_canonical_groups(records, image_manifest, rule="first_alpha")
    assert len(selections) == heldout_regen.EXPECTED_HELDOUT_PROMPTS
    # Deterministic order by prompt_id
    pids = [s.prompt_id for s in selections]
    assert pids == sorted(pids)
    # Each selection points at a real T2 entry
    for s in selections:
        assert s.cond_image_path.startswith("/fake/")
        assert s.group_id in image_manifest


def test_select_rejects_missing_t2_entry(tmp_path: pathlib.Path):
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    # Drop one group from manifest
    bad_group = sorted(image_manifest.keys())[0]
    image_manifest.pop(bad_group)
    with pytest.raises(KeyError, match=bad_group):
        heldout_regen.select_canonical_groups(records, image_manifest, rule="first_alpha")


def test_select_rejects_status_not_ok(tmp_path: pathlib.Path):
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    bad_group = sorted(image_manifest.keys())[0]
    image_manifest[bad_group]["status"] = "missing"
    with pytest.raises(ValueError, match="status != 'ok'"):
        heldout_regen.select_canonical_groups(records, image_manifest, rule="first_alpha")


# ---------- generation_config tests ----------


def test_generation_config_byte_stable():
    a = heldout_regen.canonical_generation_config(seed=42)
    b = heldout_regen.canonical_generation_config(seed=42)
    assert heldout_regen.serialize_generation_config(a) == heldout_regen.serialize_generation_config(b)


def test_generation_config_seed_diffs():
    a = heldout_regen.serialize_generation_config(heldout_regen.canonical_generation_config(seed=42))
    b = heldout_regen.serialize_generation_config(heldout_regen.canonical_generation_config(seed=43))
    assert a != b


def test_byte_identical_check_passes(tmp_path: pathlib.Path):
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=42)
    )
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "gen_config.json").write_bytes(cfg_bytes)
    (b_dir / "gen_config.json").write_bytes(cfg_bytes)
    sha = heldout_regen.assert_byte_identical_generation_configs(a_dir, b_dir)
    assert sha == hashlib.sha256(cfg_bytes).hexdigest()


def test_byte_identical_check_halts_on_drift(tmp_path: pathlib.Path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "gen_config.json").write_bytes(b'{"seed": 42}')
    (b_dir / "gen_config.json").write_bytes(b'{"seed": 43}')
    with pytest.raises(RuntimeError, match="generation_config drift"):
        heldout_regen.assert_byte_identical_generation_configs(a_dir, b_dir)


# ---------- orchestrator (dry-run) tests ----------


def test_regen_one_prompt_dry_run(tmp_path: pathlib.Path):
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=42)
    )
    sel = heldout_regen.HeldoutPrompt(
        prompt_id="abc123",
        prompt="a thing",
        group_id="g0001",
        cond_image_path="/fake/img.png",
    )
    adapter = heldout_regen.dry_run_inference_adapter()
    res = heldout_regen.regen_one_prompt(
        selection=sel,
        gen_config_bytes=cfg_bytes,
        prompt_out_root=tmp_path,
        adapter=adapter,
        ckpt_args={},
    )
    assert res["complete"] is True
    assert res["baseline"]["dry_run"] is True
    assert res["trained"]["dry_run"] is True
    assert (tmp_path / "abc123" / "baseline" / "gen_config.json").exists()
    assert (tmp_path / "abc123" / "trained" / "gen_config.json").exists()
    # gen_config bytes are identical between baseline and trained
    a = (tmp_path / "abc123" / "baseline" / "gen_config.json").read_bytes()
    b = (tmp_path / "abc123" / "trained" / "gen_config.json").read_bytes()
    assert a == b == cfg_bytes


def test_regen_one_prompt_resumes(tmp_path: pathlib.Path):
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=42)
    )
    sel = heldout_regen.HeldoutPrompt(
        prompt_id="xyz",
        prompt="prompt",
        group_id="g0",
        cond_image_path="/fake.png",
    )
    adapter = heldout_regen.dry_run_inference_adapter()
    heldout_regen.regen_one_prompt(sel, cfg_bytes, tmp_path, adapter, {})
    # Second call should short-circuit due to resume
    second = heldout_regen.regen_one_prompt(sel, cfg_bytes, tmp_path, adapter, {})
    assert second.get("resumed") is True


def test_regen_all_dry_run_42_prompts(tmp_path: pathlib.Path):
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    selections = heldout_regen.select_canonical_groups(records, image_manifest)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=1)
    )
    adapter = heldout_regen.dry_run_inference_adapter()
    out_root = tmp_path / "run"
    results = heldout_regen.regen_all(
        selections=selections,
        gen_config_bytes=cfg_bytes,
        out_root=out_root,
        adapter=adapter,
        ckpt_args={},
    )
    assert len(results) == heldout_regen.EXPECTED_HELDOUT_PROMPTS
    assert all(r["complete"] for r in results)
    # All gen_configs byte-identical
    shas = {r["gen_config_sha256"] for r in results}
    assert len(shas) == 1


def test_regen_all_world_size_sharding(tmp_path: pathlib.Path):
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    selections = heldout_regen.select_canonical_groups(records, image_manifest)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=1)
    )
    adapter = heldout_regen.dry_run_inference_adapter()
    out_root = tmp_path / "run"
    rank0 = heldout_regen.regen_all(selections, cfg_bytes, out_root, adapter, {}, rank=0, world_size=4)
    rank1 = heldout_regen.regen_all(selections, cfg_bytes, out_root, adapter, {}, rank=1, world_size=4)
    rank2 = heldout_regen.regen_all(selections, cfg_bytes, out_root, adapter, {}, rank=2, world_size=4)
    rank3 = heldout_regen.regen_all(selections, cfg_bytes, out_root, adapter, {}, rank=3, world_size=4)
    total = len(rank0) + len(rank1) + len(rank2) + len(rank3)
    assert total == heldout_regen.EXPECTED_HELDOUT_PROMPTS
    # No overlap
    seen = set()
    for batch in (rank0, rank1, rank2, rank3):
        for r in batch:
            assert r["prompt_id"] not in seen
            seen.add(r["prompt_id"])


# ---------- recipe-pin assertion ----------


def test_recipe_pin_drift_halts(tmp_path: pathlib.Path):
    rd = tmp_path / "recipes"
    rd.mkdir()
    (rd / "wan22_i2v_a14b__round2_v0.yaml").write_text("anything: 1")
    (rd / "recipe_id").write_text("0000000000000000")
    with pytest.raises(RuntimeError, match="recipe pin drift"):
        heldout_regen.assert_recipe_pin(rd, expected="6bef6e104cdd3442")


def test_recipe_pin_passes_when_consistent(tmp_path: pathlib.Path):
    rd = tmp_path / "recipes"
    rd.mkdir()
    body = b"anything: 1"
    expected = hashlib.sha256(body).hexdigest()[:16]
    (rd / "wan22_i2v_a14b__round2_v0.yaml").write_bytes(body)
    (rd / "recipe_id").write_text(expected)
    got = heldout_regen.assert_recipe_pin(rd, expected=expected)
    assert got == expected
