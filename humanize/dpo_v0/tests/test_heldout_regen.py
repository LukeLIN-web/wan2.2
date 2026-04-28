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
            "trained_lora": "/ckpt/lora.safetensors",
            "low_noise_ckpt": "/ckpt/low.safetensors",
            "upstream": "/data/Wan2.2-I2V-A14B",
            "recipe_yaml": "/recipes/r.yaml",
            "compute_envelope": "single_gpu",
        },
    )
    cmd = captured["cmd"]
    assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "baseline"
    assert "--upstream" in cmd and cmd[cmd.index("--upstream") + 1] == "/data/Wan2.2-I2V-A14B"
    assert "--low-noise-ckpt" in cmd and cmd[cmd.index("--low-noise-ckpt") + 1] == "/ckpt/low.safetensors"
    assert "--recipe-yaml" in cmd and cmd[cmd.index("--recipe-yaml") + 1] == "/recipes/r.yaml"
    assert "--compute-envelope" in cmd and cmd[cmd.index("--compute-envelope") + 1] == "single_gpu"
    assert "--prompt" in cmd
    assert "--cond-image" in cmd
    assert "--out-dir" in cmd
    assert "--gen-config-json" in cmd
    assert "--seed" in cmd
    # baseline mode takes NO --lora-adapter (inference_smoke rejects it)
    assert "--lora-adapter" not in cmd
    # legacy spec-flagged forms must NOT appear
    assert "--baseline_ckpt" not in cmd
    assert "--trained_lora" not in cmd
    assert "--cond_image" not in cmd


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
            "upstream": "/data/Wan2.2-I2V-A14B",
        },
    )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--mode") + 1] == "trained"
    assert "--lora-adapter" in cmd and cmd[cmd.index("--lora-adapter") + 1] == "/ckpt/lora.safetensors"
    assert "--baseline_ckpt" not in cmd
    assert "--trained_lora" not in cmd


def test_python_api_adapter_signature_matches_rl5_actual(tmp_path: pathlib.Path):
    """The python_api adapter must call run_one_sample with the ACTUAL
    signature in rl5's inference_smoke.py on rlcr/task-6 (rl1 caught the
    spec/impl drift in #dpo:dac89b67 msg `04170e0b`):

        mode, out_dir, upstream, torch_dtype, device, prompt, cond_image_path,
        generation_config, generation_config_bytes, lora_adapter_path,
        compute_envelope, recipe_id

    The adapter constructs UpstreamPaths(upstream_root=...) from
    ckpt_args["upstream"], resolves recipe_id via assert_recipe_pin, and
    forwards torch.bfloat16 / "cuda" defaults.
    """
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class FakeUpstreamPaths:
        upstream_root: pathlib.Path

    captured: dict = {}

    def fake_run_one_sample(
        *, mode, out_dir, upstream, torch_dtype, device, prompt, cond_image_path,
        generation_config, generation_config_bytes, lora_adapter_path,
        compute_envelope, recipe_id, high_noise_sha=None, low_noise_sha=None,
    ):
        captured.update(
            mode=mode, out_dir=out_dir, upstream=upstream,
            torch_dtype=torch_dtype, device=device,
            prompt=prompt, cond_image_path=cond_image_path,
            generation_config=generation_config,
            generation_config_bytes=generation_config_bytes,
            lora_adapter_path=lora_adapter_path,
            compute_envelope=compute_envelope,
            recipe_id=recipe_id,
        )
        return {
            "mode": mode,
            "ckpt_shas": {
                "high_noise_base_sha256": "h0",
                "low_noise_frozen_sha256": "l0",
                **({"lora_adapter_sha256": "la"} if mode == "trained" else {}),
            },
        }

    # Stub assert_recipe_pin so test doesn't need a real recipe YAML on disk.
    def fake_assert_recipe_pin(recipes_dir):
        return "6bef6e104cdd3442"

    adapter = heldout_regen.python_api_inference_adapter(
        fake_run_one_sample,
        UpstreamPaths=FakeUpstreamPaths,
        assert_recipe_pin=fake_assert_recipe_pin,
    )
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
            "trained_lora": "/ckpt/lora.safetensors",
            "low_noise_ckpt": "/ckpt/low.safetensors",
            "upstream": "/data/Wan2.2-I2V-A14B",
            "recipe_yaml": "/recipes/r.yaml",
            "compute_envelope": "multi_gpu_inference_seed_parallel",
            "device": "cuda",
        },
    )
    # Signature contract assertions
    assert captured["mode"] == "trained"
    assert captured["prompt"] == "a thing"
    # Path conversion: cond_image_path str → pathlib.Path
    assert isinstance(captured["cond_image_path"], pathlib.Path)
    assert str(captured["cond_image_path"]) == "/fake/img.png"
    # generation_config dict + generation_config_bytes paired
    assert captured["generation_config"]["seed"] == 11
    assert captured["generation_config_bytes"] == cfg_bytes
    # UpstreamPaths constructed with Path
    assert isinstance(captured["upstream"], FakeUpstreamPaths)
    assert captured["upstream"].upstream_root == pathlib.Path("/data/Wan2.2-I2V-A14B")
    # lora_adapter_path is Path for trained, None for baseline
    assert captured["lora_adapter_path"] == pathlib.Path("/ckpt/lora.safetensors")
    assert captured["compute_envelope"] == "multi_gpu_inference_seed_parallel"
    assert captured["recipe_id"] == "6bef6e104cdd3442"
    assert captured["device"] == "cuda"
    # torch_dtype resolved from gen_config dtype string
    import torch
    assert captured["torch_dtype"] is torch.bfloat16
    # ckpt_shas propagated from inner result
    assert out["ckpt_shas"] == {
        "high_noise_base_sha256": "h0",
        "low_noise_frozen_sha256": "l0",
        "lora_adapter_sha256": "la",
    }


def test_python_api_adapter_against_inference_smoke_real_signature():
    """Cross-module signature integration test (BL-20260428-mock-not-import-
    catches-spec-drift): verify the adapter passes EXACTLY the kwargs
    rl5's actual ``run_one_sample`` accepts.

    Uses ``inspect.signature`` on the real import — no GPU exec, no
    diffsynth dependency, no torch model load. If a future commit
    renames a kwarg or drops a parameter on either side, this test
    flags it at test-time instead of M6-launch-time. The mock-only
    ``test_python_api_adapter_signature_matches_rl5_actual`` above
    encodes the orchestrator's call kwargs; this test verifies they
    match the callee's actual parameter set.
    """
    import inspect
    try:
        from inference_smoke import run_one_sample  # type: ignore
    except ImportError as exc:
        pytest.skip(f"inference_smoke not importable in this env: {exc}")

    sig = inspect.signature(run_one_sample)
    callee_param_names = set(sig.parameters.keys())

    # Kwargs the orchestrator's python_api_inference_adapter passes
    # (kept in sync with the adapter implementation manually — drift
    # between this set and the adapter call site would make the
    # adapter pass an undefined kwarg).
    adapter_kwargs = {
        "mode",
        "out_dir",
        "upstream",
        "torch_dtype",
        "device",
        "prompt",
        "cond_image_path",
        "generation_config",
        "generation_config_bytes",
        "lora_adapter_path",
        "compute_envelope",
        "recipe_id",
    }

    missing_in_callee = adapter_kwargs - callee_param_names
    assert not missing_in_callee, (
        f"orchestrator passes kwargs that don't exist in run_one_sample: "
        f"{sorted(missing_in_callee)}. signature drift between "
        f"heldout_regen.python_api_inference_adapter and "
        f"inference_smoke.run_one_sample. expected callee params: "
        f"{sorted(callee_param_names)}"
    )

    # Optional params on callee (e.g. high_noise_sha / low_noise_sha cache)
    # are fine — adapter just doesn't use them. Required params on callee
    # that adapter DOESN'T pass would TypeError at call time:
    callee_required = {
        name for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL,
                               inspect.Parameter.VAR_KEYWORD)
    }
    missing_in_adapter = callee_required - adapter_kwargs
    assert not missing_in_adapter, (
        f"run_one_sample requires kwargs the adapter doesn't pass: "
        f"{sorted(missing_in_adapter)}. signature drift in the other "
        f"direction. orchestrator passes: {sorted(adapter_kwargs)}"
    )


def test_python_api_adapter_baseline_passes_none_lora(tmp_path: pathlib.Path):
    """Baseline mode must pass lora_adapter_path=None to run_one_sample."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class FakeUpstreamPaths:
        upstream_root: pathlib.Path

    captured: dict = {}

    def fake_run_one_sample(*, mode, lora_adapter_path, **kw):
        captured["mode"] = mode
        captured["lora_adapter_path"] = lora_adapter_path
        return {"mode": mode, "ckpt_shas": {"high_noise_base_sha256": "x"}}

    adapter = heldout_regen.python_api_inference_adapter(
        fake_run_one_sample,
        UpstreamPaths=FakeUpstreamPaths,
        assert_recipe_pin=lambda d: "6bef6e104cdd3442",
    )
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    adapter(
        run_kind="baseline",
        prompt="x",
        cond_image_path="/fake.png",
        gen_config_bytes=cfg_bytes,
        out_dir=tmp_path / "out_baseline",
        ckpt_args={
            "trained_lora": "/ckpt/lora.safetensors",  # ignored in baseline
            "upstream": "/data/Wan2.2-I2V-A14B",
        },
    )
    assert captured["mode"] == "baseline"
    assert captured["lora_adapter_path"] is None


def test_subprocess_adapter_requires_upstream(tmp_path: pathlib.Path):
    """upstream is the only orchestrator-level required ckpt_arg for the
    subprocess adapter; everything else is optional or mode-conditional.
    Baseline does NOT need a baseline_ckpt — inference_smoke baseline runs
    off the canonical sharded high_noise expert under <upstream>/."""
    fake_inference = tmp_path / "inference_smoke.py"
    fake_inference.write_text("# stub")
    adapter = heldout_regen.subprocess_inference_adapter(fake_inference)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=7)
    )
    with pytest.raises(KeyError, match="upstream"):
        adapter(
            run_kind="baseline",
            prompt="x",
            cond_image_path="/fake.png",
            gen_config_bytes=cfg_bytes,
            out_dir=tmp_path / "out",
            ckpt_args={"trained_lora": "/ckpt/lora.safetensors"},  # missing upstream
        )


def test_subprocess_adapter_trained_requires_lora(tmp_path: pathlib.Path):
    fake_inference = tmp_path / "inference_smoke.py"
    fake_inference.write_text("# stub")
    adapter = heldout_regen.subprocess_inference_adapter(fake_inference)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=7)
    )
    with pytest.raises(KeyError, match="trained_lora"):
        adapter(
            run_kind="trained",
            prompt="x",
            cond_image_path="/fake.png",
            gen_config_bytes=cfg_bytes,
            out_dir=tmp_path / "out",
            ckpt_args={"upstream": "/data/Wan2.2-I2V-A14B"},  # missing trained_lora
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
        out_idx = cmd.index("--out-dir") + 1
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
            "upstream": "/data/Wan2.2-I2V-A14B",
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


# ---------- pipe-cache + mode-batched (round-4 N>>1 calls/rank) ----------


import dataclasses


def _build_pipe_cache_adapter(captured: list[dict]):
    """Build a python_api adapter wired to fake build_pipeline + run_one_sample.

    ``captured`` is appended one record per run_one_sample call so tests can
    assert the (pipe-identity, lora_already_attached, lora_sha) wiring.
    """

    @dataclasses.dataclass(frozen=True)
    class FakeUpstreamPaths:
        upstream_root: pathlib.Path

    class FakePipe:
        """Sentinel returned by build_pipeline; identity-tracked across calls."""

        _next = 0

        def __init__(self):
            FakePipe._next += 1
            self.id = FakePipe._next

    build_count = {"n": 0}

    def fake_build_pipeline(upstream, *, torch_dtype, device):
        build_count["n"] += 1
        return FakePipe()

    def fake_run_one_sample(
        *,
        mode,
        out_dir,
        upstream,
        torch_dtype,
        device,
        prompt,
        cond_image_path,
        generation_config,
        generation_config_bytes,
        lora_adapter_path,
        compute_envelope,
        recipe_id,
        high_noise_sha=None,
        low_noise_sha=None,
        pipe=None,
        lora_already_attached=False,
        lora_sha=None,
    ):
        captured.append({
            "mode": mode,
            "prompt": prompt,
            "pipe_id": pipe.id if pipe is not None else None,
            "lora_already_attached": lora_already_attached,
            "lora_sha_param": lora_sha,
            "lora_adapter_path": lora_adapter_path,
            "high_noise_sha": high_noise_sha,
        })
        return {
            "mode": mode,
            "ckpt_shas": {
                "high_noise_base_sha256": "h0",
                "low_noise_frozen_sha256": "l0",
                **({"lora_adapter_sha256": "la"} if mode == "trained" else {}),
            },
        }

    adapter = heldout_regen.python_api_inference_adapter(
        fake_run_one_sample,
        UpstreamPaths=FakeUpstreamPaths,
        assert_recipe_pin=lambda rd: "6bef6e104cdd3442",
        cache_pipe=True,
        build_pipeline=fake_build_pipeline,
    )
    return adapter, build_count, FakePipe


def _ckpt_args_for_pipe_cache():
    return {
        "trained_lora": "/ckpt/lora.safetensors",
        "upstream": "/data/Wan2.2-I2V-A14B",
        "recipe_yaml": "/recipes/r.yaml",
        "compute_envelope": "multi_gpu_inference_seed_parallel",
        "device": "cuda",
    }


def test_python_api_cache_pipe_builds_once_across_calls(tmp_path: pathlib.Path):
    captured: list[dict] = []
    adapter, build_count, _ = _build_pipe_cache_adapter(captured)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    ckpt_args = _ckpt_args_for_pipe_cache()

    # 5 baselines, then 5 trained — mode-batched order required by cache.
    for i in range(5):
        adapter("baseline", f"prompt {i}", "/img.png", cfg_bytes,
                tmp_path / f"b{i}", ckpt_args)
    for i in range(5):
        adapter("trained", f"prompt {i}", "/img.png", cfg_bytes,
                tmp_path / f"t{i}", ckpt_args)

    # Single build_pipeline across all 10 calls (the whole point).
    assert build_count["n"] == 1, f"expected 1 build, got {build_count['n']}"
    # All 10 calls saw the same pipe identity.
    pipe_ids = {c["pipe_id"] for c in captured}
    assert pipe_ids == {1}
    # Baseline calls: lora_already_attached=False
    assert all(c["lora_already_attached"] is False for c in captured if c["mode"] == "baseline")
    # First trained call: not yet attached → lora_already_attached=False
    trained = [c for c in captured if c["mode"] == "trained"]
    assert trained[0]["lora_already_attached"] is False
    # Subsequent trained calls reuse: lora_already_attached=True + cached lora_sha
    for t in trained[1:]:
        assert t["lora_already_attached"] is True
        assert t["lora_sha_param"] == "la"
    # Sharded SHAs cached after first call (passed to run_one_sample on subsequent).
    assert captured[0]["high_noise_sha"] is None
    assert all(c["high_noise_sha"] == "h0" for c in captured[1:])


def test_python_api_cache_pipe_rejects_baseline_after_trained(tmp_path: pathlib.Path):
    captured: list[dict] = []
    adapter, _, _ = _build_pipe_cache_adapter(captured)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    ckpt_args = _ckpt_args_for_pipe_cache()

    adapter("baseline", "p0", "/img.png", cfg_bytes, tmp_path / "b0", ckpt_args)
    adapter("trained", "p0", "/img.png", cfg_bytes, tmp_path / "t0", ckpt_args)
    with pytest.raises(RuntimeError, match="baseline run after trained"):
        adapter("baseline", "p1", "/img.png", cfg_bytes, tmp_path / "b1", ckpt_args)


def test_python_api_cache_pipe_rejects_different_lora_paths(tmp_path: pathlib.Path):
    captured: list[dict] = []
    adapter, _, _ = _build_pipe_cache_adapter(captured)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    ckpt_args = _ckpt_args_for_pipe_cache()

    adapter("trained", "p0", "/img.png", cfg_bytes, tmp_path / "t0", ckpt_args)
    ckpt_args2 = dict(ckpt_args)
    ckpt_args2["trained_lora"] = "/ckpt/other.safetensors"
    with pytest.raises(RuntimeError, match="Different LoRA paths"):
        adapter("trained", "p1", "/img.png", cfg_bytes, tmp_path / "t1", ckpt_args2)


def test_python_api_cache_pipe_rejects_upstream_change(tmp_path: pathlib.Path):
    captured: list[dict] = []
    adapter, _, _ = _build_pipe_cache_adapter(captured)
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    ckpt_args = _ckpt_args_for_pipe_cache()

    adapter("baseline", "p0", "/img.png", cfg_bytes, tmp_path / "b0", ckpt_args)
    ckpt_args2 = dict(ckpt_args)
    ckpt_args2["upstream"] = "/data/different/Wan2.2"
    with pytest.raises(RuntimeError, match="upstream_root"):
        adapter("baseline", "p1", "/img.png", cfg_bytes, tmp_path / "b1", ckpt_args2)


def test_python_api_no_cache_pipe_omits_pipe_kwarg(tmp_path: pathlib.Path):
    """Backward compat: when cache_pipe=False, run_one_sample receives no
    pipe= kwarg (so it builds its own pipeline as before)."""

    @dataclasses.dataclass(frozen=True)
    class FakeUpstreamPaths:
        upstream_root: pathlib.Path

    captured: dict = {}

    def fake_run_one_sample(
        *, mode, out_dir, upstream, torch_dtype, device, prompt, cond_image_path,
        generation_config, generation_config_bytes, lora_adapter_path,
        compute_envelope, recipe_id, high_noise_sha=None, low_noise_sha=None,
        **extras,
    ):
        captured["extras"] = extras
        return {"mode": mode, "ckpt_shas": {}}

    adapter = heldout_regen.python_api_inference_adapter(
        fake_run_one_sample,
        UpstreamPaths=FakeUpstreamPaths,
        assert_recipe_pin=lambda rd: "6bef6e104cdd3442",
        # cache_pipe defaults to False
    )
    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    adapter("baseline", "p0", "/img.png", cfg_bytes, tmp_path / "b0",
            _ckpt_args_for_pipe_cache())
    # No pipe / lora_already_attached / lora_sha kwargs leaked through.
    assert captured["extras"] == {}


def test_regen_all_mode_batched_runs_baselines_first(tmp_path: pathlib.Path):
    """With --mode-batched, per rank: all baselines first, then all trained.

    Verifies the call order at the adapter boundary so cache_pipe + LoRA
    state machine is fed the right sequence.
    """
    root = _build_heldout_fixture(tmp_path)
    selections = heldout_regen.select_canonical_groups(
        heldout_regen.load_heldout_records(root),
        heldout_regen.load_t2_image_manifest(root),
    )
    selections = selections[:4]  # smoke scope

    call_order: list[tuple[str, str]] = []

    def adapter(run_kind, prompt, cond_image_path, gen_config_bytes, out_dir, ckpt_args):
        out_dir.mkdir(parents=True, exist_ok=True)
        call_order.append((run_kind, prompt))
        gen_cfg_path = out_dir / "gen_config.json"
        gen_cfg_path.write_bytes(gen_config_bytes)
        return {
            "run_kind": run_kind,
            "out_dir": str(out_dir),
            "wall_seconds": 0.01,
            "result": {"mode": run_kind, "ckpt_shas": {}},
            "ckpt_shas": {},
        }

    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    out_root = tmp_path / "out"
    out_root.mkdir()

    results = heldout_regen.regen_all(
        selections=selections,
        gen_config_bytes=cfg_bytes,
        out_root=out_root,
        adapter=adapter,
        ckpt_args={},
        resume=False,
        rank=0,
        world_size=1,
        mode_batched=True,
    )
    # All baselines first, then all trained.
    kinds = [k for k, _ in call_order]
    assert kinds == ["baseline"] * 4 + ["trained"] * 4
    assert len(results) == 4
    # Per-prompt summaries written.
    for sel in selections:
        summary = out_root / "heldout_regen" / sel.prompt_id / "prompt_manifest.json"
        assert summary.exists()
        rec = json.loads(summary.read_bytes())
        assert rec["complete"] is True


def test_regen_all_mode_batched_resumes_completed_prompts(tmp_path: pathlib.Path):
    """A prompt with an existing complete=True summary skips both passes."""
    root = _build_heldout_fixture(tmp_path)
    selections = heldout_regen.select_canonical_groups(
        heldout_regen.load_heldout_records(root),
        heldout_regen.load_t2_image_manifest(root),
    )
    selections = selections[:3]

    out_root = tmp_path / "out"
    (out_root / "heldout_regen").mkdir(parents=True)
    # Pre-populate prompt[0] as complete.
    pre = out_root / "heldout_regen" / selections[0].prompt_id
    pre.mkdir()
    (pre / "prompt_manifest.json").write_text(json.dumps({
        "prompt_id": selections[0].prompt_id,
        "complete": True,
    }))

    call_order: list[str] = []

    def adapter(run_kind, prompt, cond_image_path, gen_config_bytes, out_dir, ckpt_args):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gen_config.json").write_bytes(gen_config_bytes)
        call_order.append(run_kind)
        return {"run_kind": run_kind, "ckpt_shas": {}}

    cfg_bytes = heldout_regen.serialize_generation_config(
        heldout_regen.canonical_generation_config(seed=11)
    )
    results = heldout_regen.regen_all(
        selections=selections,
        gen_config_bytes=cfg_bytes,
        out_root=out_root,
        adapter=adapter,
        ckpt_args={},
        resume=True,
        rank=0,
        world_size=1,
        mode_batched=True,
    )
    # Prompt 0 resumed → 0 calls; prompts 1, 2 → 2 baselines + 2 trained.
    assert call_order == ["baseline", "baseline", "trained", "trained"]
    assert len(results) == 3
    assert any(r.get("resumed") is True for r in results)


def test_run_one_sample_pipe_param_skips_build(tmp_path: pathlib.Path):
    """When pipe=<sentinel> is passed, run_one_sample skips build_pipeline.

    Inject fake ``diffsynth.utils.data`` into sys.modules before importing
    inference_smoke so the test runs on hosts without DiffSynth installed
    (the orchestrator dev box). The lazy ``from diffsynth.utils.data
    import save_video`` inside run_one_sample picks up our fake.
    """
    import types
    fake_du = types.ModuleType("diffsynth.utils.data")
    fake_du.save_video = lambda video, path, fps=1, quality=5: pathlib.Path(path).write_bytes(b"\x00")
    fake_u = types.ModuleType("diffsynth.utils")
    fake_u.data = fake_du
    fake_diffsynth = sys.modules.setdefault("diffsynth", types.ModuleType("diffsynth"))
    sys.modules.setdefault("diffsynth.utils", fake_u)
    sys.modules.setdefault("diffsynth.utils.data", fake_du)

    inference_smoke = pytest.importorskip("inference_smoke")
    torch = pytest.importorskip("torch")

    class FakePipe:
        def __call__(self, **kw):
            class _V:
                def cpu(self): return self
            return _V()

    build_called = {"n": 0}
    attach_called = {"n": 0}

    def fake_build_pipeline(upstream, *, torch_dtype, device):
        build_called["n"] += 1
        return FakePipe()

    def fake_attach_lora(pipe, lora_path):
        attach_called["n"] += 1

    import unittest.mock as mock
    with mock.patch.object(inference_smoke, "build_pipeline", fake_build_pipeline), \
         mock.patch.object(inference_smoke, "attach_lora", fake_attach_lora), \
         mock.patch.object(inference_smoke, "sharded_ckpt_sha", lambda paths: "z" * 64), \
         mock.patch.object(inference_smoke, "file_sha256", lambda p: "f" * 64):
        cfg = inference_smoke.build_generation_config(
            type("A", (), dict(
                seed=1, num_inference_steps=1, cfg_scale=1.0, negative_prompt="",
                height=64, width=64, num_frames=1, fps=1,
                torch_dtype="bf16", switch_DiT_boundary=875,
            ))()
        )
        cfg_bytes = inference_smoke.serialize_generation_config(cfg)

        cached_pipe = FakePipe()
        upstream = inference_smoke.UpstreamPaths(upstream_root=tmp_path / "wan")
        cond = tmp_path / "cond.png"
        from PIL import Image
        Image.new("RGB", (64, 64)).save(cond)

        manifest = inference_smoke.run_one_sample(
            mode="baseline",
            out_dir=tmp_path / "out",
            upstream=upstream,
            torch_dtype=torch.bfloat16,
            device="cpu",
            prompt="a thing",
            cond_image_path=cond,
            generation_config=cfg,
            generation_config_bytes=cfg_bytes,
            lora_adapter_path=None,
            compute_envelope="single_gpu",
            recipe_id="6bef6e104cdd3442",
            pipe=cached_pipe,
        )

    assert build_called["n"] == 0, "build_pipeline must be skipped when pipe= is passed"
    assert attach_called["n"] == 0
    assert manifest["mode"] == "baseline"
