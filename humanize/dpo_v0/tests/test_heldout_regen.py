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


def test_limit_prompts_slices_deterministic(tmp_path: pathlib.Path):
    """--limit-prompts N takes the first N selections in deterministic
    prompt_id order — the same N prompts every run, regardless of rank
    sharding applied later."""
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    selections = heldout_regen.select_canonical_groups(records, image_manifest)
    assert len(selections) == heldout_regen.EXPECTED_HELDOUT_PROMPTS

    sliced = selections[:8]
    assert len(sliced) == 8
    pids = [s.prompt_id for s in sliced]
    assert pids == sorted(pids)
    sliced_again = heldout_regen.select_canonical_groups(records, image_manifest)[:8]
    assert [s.prompt_id for s in sliced] == [s.prompt_id for s in sliced_again]


def test_limit_prompts_with_world_size_sharding(tmp_path: pathlib.Path):
    """--limit-prompts + --world-size compose: limit first, then shard."""
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    selections = heldout_regen.select_canonical_groups(records, image_manifest)[:8]
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
    assert total == 8
    seen = set()
    for batch in (rank0, rank1, rank2, rank3):
        for r in batch:
            assert r["prompt_id"] not in seen
            seen.add(r["prompt_id"])


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


# ---------- subprocess adapter CLI shape (rl5 lock 5e8993eb) ----------


def test_subprocess_adapter_emits_locked_cli(tmp_path: pathlib.Path, monkeypatch):
    """Verify the subprocess command shape matches rl5's locked inference_smoke.py CLI."""
    captured: dict = {}

    class _StubResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = list(cmd)
        return _StubResult()

    monkeypatch.setattr(heldout_regen.subprocess, "run", fake_run)

    fake_inference = tmp_path / "inference_smoke.py"
    fake_inference.write_text("# stub")
    adapter = heldout_regen.subprocess_inference_adapter(fake_inference)

    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=7)
    )
    adapter(
        run_kind="baseline",
        prompt="a thing happens",
        cond_image_path="/fake/img.png",
        gen_config_bytes=cfg_bytes,
        out_dir=tmp_path / "out_baseline",
        ckpt_args={
            "baseline_ckpt": "/ckpt/base.safetensors",
            "trained_lora": "/ckpt/lora.safetensors",
            "low_noise_ckpt": "/ckpt/low.safetensors",
            "upstream": "/data/Wan2.2-I2V-A14B",
            "recipe_yaml": "/recipes/r.yaml",
        },
    )
    cmd = captured["cmd"]
    assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "baseline"
    assert "--baseline_ckpt" in cmd and cmd[cmd.index("--baseline_ckpt") + 1] == "/ckpt/base.safetensors"
    assert "--low_noise_ckpt" in cmd and cmd[cmd.index("--low_noise_ckpt") + 1] == "/ckpt/low.safetensors"
    assert "--upstream" in cmd and cmd[cmd.index("--upstream") + 1] == "/data/Wan2.2-I2V-A14B"
    assert "--recipe_yaml" in cmd and cmd[cmd.index("--recipe_yaml") + 1] == "/recipes/r.yaml"
    assert "--prompt" in cmd
    assert "--cond_image" in cmd
    assert "--out_dir" in cmd
    assert "--gen_config_json" in cmd
    assert "--seed" in cmd
    # baseline mode does NOT pass --trained_lora
    assert "--trained_lora" not in cmd


def test_subprocess_adapter_trained_mode(tmp_path: pathlib.Path, monkeypatch):
    captured: dict = {}

    class _StubResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        heldout_regen.subprocess, "run",
        lambda cmd, capture_output, text, timeout: (captured.update({"cmd": list(cmd)}) or _StubResult()),
    )

    fake_inference = tmp_path / "inference_smoke.py"
    fake_inference.write_text("# stub")
    adapter = heldout_regen.subprocess_inference_adapter(fake_inference)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=7)
    )
    adapter(
        run_kind="trained",
        prompt="x",
        cond_image_path="/fake.png",
        gen_config_bytes=cfg_bytes,
        out_dir=tmp_path / "out_trained",
        ckpt_args={
            "trained_lora": "/ckpt/lora.safetensors",
            "low_noise_ckpt": "/ckpt/low.safetensors",
        },
    )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--mode") + 1] == "trained"
    assert "--trained_lora" in cmd
    assert "--baseline_ckpt" not in cmd  # trained mode doesn't need baseline


def test_python_api_adapter_signature_matches_rl5_lock(tmp_path: pathlib.Path):
    """The python_api adapter must call run_one_sample with the keyword arg
    set rl5 locked at msg `5e8993eb` / `ac00949`:

        mode, prompt, cond_image, gen_config, out_dir, seed,
        baseline_ckpt, trained_lora, low_noise_ckpt, upstream, recipe_yaml

    Any drift in this signature would break the wire-in. This test mocks
    run_one_sample as a kwargs-only callable to assert the contract.
    """
    captured: dict = {}

    def fake_run_one_sample(*, mode, prompt, cond_image, gen_config, out_dir, seed,
                            baseline_ckpt=None, trained_lora=None, low_noise_ckpt=None,
                            upstream=None, recipe_yaml=None):
        captured.update(
            mode=mode, prompt=prompt, cond_image=cond_image, gen_config=gen_config,
            out_dir=out_dir, seed=seed,
            baseline_ckpt=baseline_ckpt, trained_lora=trained_lora,
            low_noise_ckpt=low_noise_ckpt, upstream=upstream, recipe_yaml=recipe_yaml,
        )
        return {
            "mode": mode,
            "ckpt_shas": {
                "high_noise_base_sha256": "h0",
                "low_noise_frozen_sha256": "l0",
                **({"lora_adapter_sha256": "la"} if mode == "trained" else {}),
            },
        }

    adapter = heldout_regen.python_api_inference_adapter(fake_run_one_sample)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    out = adapter(
        run_kind="trained",
        prompt="a thing",
        cond_image_path="/fake/img.png",
        gen_config_bytes=cfg_bytes,
        out_dir=tmp_path / "out",
        ckpt_args={
            "baseline_ckpt": "/ckpt/base.safetensors",
            "trained_lora": "/ckpt/lora.safetensors",
            "low_noise_ckpt": "/ckpt/low.safetensors",
            "upstream": "/data/Wan2.2-I2V-A14B",
            "recipe_yaml": "/recipes/r.yaml",
        },
    )
    # Signature contract assertions
    assert captured["mode"] == "trained"
    assert captured["prompt"] == "a thing"
    assert captured["cond_image"] == "/fake/img.png"
    assert captured["seed"] == 11
    assert captured["baseline_ckpt"] == "/ckpt/base.safetensors"
    assert captured["trained_lora"] == "/ckpt/lora.safetensors"
    assert captured["low_noise_ckpt"] == "/ckpt/low.safetensors"
    assert captured["upstream"] == "/data/Wan2.2-I2V-A14B"
    assert captured["recipe_yaml"] == "/recipes/r.yaml"
    # ckpt_shas propagated from inner result
    assert out["ckpt_shas"] == {
        "high_noise_base_sha256": "h0",
        "low_noise_frozen_sha256": "l0",
        "lora_adapter_sha256": "la",
    }


def test_subprocess_adapter_requires_low_noise(tmp_path: pathlib.Path):
    fake_inference = tmp_path / "inference_smoke.py"
    fake_inference.write_text("# stub")
    adapter = heldout_regen.subprocess_inference_adapter(fake_inference)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=7)
    )
    with pytest.raises(KeyError, match="low_noise_ckpt"):
        adapter(
            run_kind="baseline",
            prompt="x",
            cond_image_path="/fake.png",
            gen_config_bytes=cfg_bytes,
            out_dir=tmp_path / "out",
            ckpt_args={"baseline_ckpt": "/ckpt/base.safetensors"},
        )


# ---------- ckpt_shas propagation (rl5 schema a76f4c8e) ----------


def test_extract_ckpt_shas_reads_manifest(tmp_path: pathlib.Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({
        "ckpt_shas": {
            "high_noise_base_sha256": "deadbeef",
            "low_noise_frozen_sha256": "cafef00d",
            "lora_adapter_sha256": "12345678",
        },
    }))
    shas = heldout_regen._extract_ckpt_shas(out_dir)
    assert shas == {
        "high_noise_base_sha256": "deadbeef",
        "low_noise_frozen_sha256": "cafef00d",
        "lora_adapter_sha256": "12345678",
    }


def test_extract_ckpt_shas_missing_manifest(tmp_path: pathlib.Path):
    assert heldout_regen._extract_ckpt_shas(tmp_path) is None


def test_extract_ckpt_shas_run_manifest_alias(tmp_path: pathlib.Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "ckpt_shas": {"high_noise_base_sha256": "abc"},
    }))
    shas = heldout_regen._extract_ckpt_shas(out_dir)
    assert shas == {"high_noise_base_sha256": "abc"}


def test_subprocess_adapter_propagates_ckpt_shas(tmp_path: pathlib.Path, monkeypatch):
    """When inference_smoke writes manifest.json with ckpt_shas, adapter surfaces it."""
    captured: dict = {}

    class _StubResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = list(cmd)
        # Simulate inference_smoke dropping a manifest with ckpt_shas in out_dir.
        out_idx = cmd.index("--out_dir") + 1
        out_dir = pathlib.Path(cmd[out_idx])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps({
            "ckpt_shas": {
                "high_noise_base_sha256": "aaaaa",
                "low_noise_frozen_sha256": "bbbbb",
            },
        }))
        return _StubResult()

    monkeypatch.setattr(heldout_regen.subprocess, "run", fake_run)

    fake_inference = tmp_path / "inference_smoke.py"
    fake_inference.write_text("# stub")
    adapter = heldout_regen.subprocess_inference_adapter(fake_inference)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=7)
    )
    out = adapter(
        run_kind="baseline",
        prompt="x",
        cond_image_path="/fake.png",
        gen_config_bytes=cfg_bytes,
        out_dir=tmp_path / "out",
        ckpt_args={
            "baseline_ckpt": "/ckpt/base.safetensors",
            "low_noise_ckpt": "/ckpt/low.safetensors",
        },
    )
    assert out["ckpt_shas"] == {
        "high_noise_base_sha256": "aaaaa",
        "low_noise_frozen_sha256": "bbbbb",
    }


# ---------- cond-image fallback root (rl1 0030832 mirror) ----------


def test_resolve_cond_image_fallback_root(tmp_path: pathlib.Path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "img.png").write_bytes(b"\x89PNG")
    primary_missing = "/this/path/does/not/exist/img.png"
    out = heldout_regen._resolve_cond_image_path(primary_missing, fallback)
    assert out == str(fallback / "img.png")


def test_resolve_cond_image_passthrough_when_primary_exists(tmp_path: pathlib.Path):
    primary_real = tmp_path / "real.png"
    primary_real.write_bytes(b"x")
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    out = heldout_regen._resolve_cond_image_path(str(primary_real), fallback)
    assert out == str(primary_real)


def test_resolve_cond_image_returns_primary_when_fallback_misses(tmp_path: pathlib.Path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    primary_missing = "/no/such/img.png"
    out = heldout_regen._resolve_cond_image_path(primary_missing, fallback)
    assert out == primary_missing  # unchanged so downstream loader fails loudly


def test_select_canonical_with_fallback(tmp_path: pathlib.Path):
    root = _build_heldout_fixture(tmp_path)
    records = heldout_regen.load_heldout_records(root)
    image_manifest = heldout_regen.load_t2_image_manifest(root)
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    # Materialize one fallback for the first canonical group.
    first_group_id = sorted(image_manifest.keys())[0]
    (fallback / f"{first_group_id}.png").write_bytes(b"\x89PNG")
    selections = heldout_regen.select_canonical_groups(
        records, image_manifest, rule="first_alpha",
        cond_image_fallback_root=fallback,
    )
    # The selection that maps to first_group_id should resolve via fallback.
    matches = [s for s in selections if s.group_id == first_group_id]
    assert len(matches) == 1
    assert str(fallback) in matches[0].cond_image_path


# ---------- recipe-pin assertion ----------


def test_recipe_pin_drift_halts(tmp_path: pathlib.Path):
    rd = tmp_path / "recipes"
    rd.mkdir()
    (rd / "wan22_i2v_a14b__round2_v0.yaml").write_text("anything: 1")
    (rd / "recipe_id").write_text("0000000000000000")
    # Underlying impl is manifest_writer.assert_recipe_pins which raises
    # AssertionError (3-way assert with all three values inline).
    with pytest.raises(AssertionError, match="recipe pin drift"):
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
