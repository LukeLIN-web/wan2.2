"""M6 heldout regeneration orchestration scaffold.

Reads ``<T0_T3_ROOT>/splits/heldout.json`` (579 pairs / 245 groups / 42
unique prompts) and ``<T0_T3_ROOT>/t2/image_manifest.json`` (1440 group
entries), picks one canonical (group, image) per prompt under a
deterministic rule, and runs ``inference_smoke.py`` twice per prompt
(baseline ckpt + trained ckpt) under one byte-identical generation
config.

This is the M6 orchestration scaffold. The actual generation work is
delegated to ``inference_smoke.py`` (rl5, task #7) — this module
calls it via subprocess CLI (preferred for byte-identical config + a
clean process boundary) or via Python API hook once rl5 exposes
``inference_smoke.run_one_sample(...)``.

Plan: humanize/i2v.md AC-7.1, AC-7.2, AC-7.3, M6, task8.

Hard contracts honoured here:
  - Heldout is NEVER pre-encoded. We invoke inference, which encodes
    inside its own pipeline; we do not touch VAE-encoded latents for
    heldout videos.
  - generation_config is byte-identical between baseline and trained:
    we serialize a single dict object once via canonical JSON and pass
    that exact bytes to both invocations; any divergence is a hard
    fail.
  - heldout conditioning images come from <T0_T3_ROOT>/t2/image_manifest.json
    (canonical T2 mapping). We do NOT re-resolve at eval time.
  - 42 unique prompts in heldout.json is asserted at startup (AC-7.1
    is "the 42 heldout prompts"; the round-0 audit confirms 42 unique
    prompts under 245 groups under 579 pairs).
  - recipe_id 6bef6e104cdd3442 (AC-3.1) is asserted before any
    generation call.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional


HERE = pathlib.Path(__file__).resolve().parent

EXPECTED_RECIPE_ID = "6bef6e104cdd3442"  # AC-3.1, frozen in task-6 round 1
EXPECTED_HELDOUT_PROMPTS = 42  # round-0 audit §3
EXPECTED_HELDOUT_GROUPS = 245
EXPECTED_HELDOUT_PAIRS = 579


# ---------- canonical artifact loaders ----------


def assert_recipe_pin(recipes_dir: pathlib.Path, expected: str = EXPECTED_RECIPE_ID) -> str:
    yaml_path = recipes_dir / "wan22_i2v_a14b__round2_v0.yaml"
    pin_path = recipes_dir / "recipe_id"
    if not yaml_path.exists() or not pin_path.exists():
        raise FileNotFoundError(
            f"recipe artifacts missing: yaml={yaml_path.exists()}, pin={pin_path.exists()} "
            f"(under {recipes_dir})"
        )
    fresh = hashlib.sha256(yaml_path.read_bytes()).hexdigest()[:16]
    on_disk = pin_path.read_text(encoding="ascii").strip()
    if not (fresh == on_disk == expected):
        raise RuntimeError(
            f"recipe pin drift: fresh={fresh}, on_disk={on_disk}, expected={expected}"
        )
    return on_disk


def load_heldout_records(t0_t3_root: pathlib.Path) -> list[dict]:
    path = t0_t3_root / "splits" / "heldout.json"
    if not path.exists():
        raise FileNotFoundError(f"heldout split missing: {path}")
    records = json.loads(path.read_bytes())
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a JSON array")
    if len(records) != EXPECTED_HELDOUT_PAIRS:
        raise ValueError(
            f"heldout pair count mismatch: got {len(records)}, expected {EXPECTED_HELDOUT_PAIRS} "
            f"(round-0 audit §3); refusing to proceed without explicit re-pin sign-off."
        )
    n_prompts = len({r["prompt"] for r in records})
    n_groups = len({r["group_id"] for r in records})
    if n_prompts != EXPECTED_HELDOUT_PROMPTS:
        raise ValueError(
            f"heldout unique-prompt count mismatch: got {n_prompts}, expected {EXPECTED_HELDOUT_PROMPTS}"
        )
    if n_groups != EXPECTED_HELDOUT_GROUPS:
        raise ValueError(
            f"heldout unique-group count mismatch: got {n_groups}, expected {EXPECTED_HELDOUT_GROUPS}"
        )
    return records


def load_t2_image_manifest(t0_t3_root: pathlib.Path) -> dict:
    path = t0_t3_root / "t2" / "image_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"T2 image manifest missing: {path}")
    manifest = json.loads(path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} is not a JSON object")
    return manifest


# ---------- prompt -> canonical (group, image) selection ----------


@dataclasses.dataclass(frozen=True)
class HeldoutPrompt:
    prompt_id: str           # short stable id (sha256(prompt)[:12])
    prompt: str
    group_id: str            # canonical group chosen for this prompt
    cond_image_path: str     # canonical T2 image for that group


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def select_canonical_groups(
    records: list[dict],
    image_manifest: dict,
    rule: str = "first_alpha",
) -> list[HeldoutPrompt]:
    """For each unique prompt, pick one canonical group + its T2 image.

    ``rule`` options:
      - ``first_alpha`` (default): for each prompt, pick the group_id
        sorted alphabetically first. Deterministic, no extra inputs.
      - ``first_in_record_order``: pick the first group_id encountered
        when iterating records in their on-disk order. Deterministic
        only if the upstream record order is canonical; we sort
        records by ``pair_id`` first to get reproducibility.

    Returns a list sorted by ``prompt_id`` so downstream loops are
    deterministic and CI-stable regardless of dict iteration order.
    """
    groups_by_prompt: dict[str, set[str]] = {}
    if rule == "first_in_record_order":
        sorted_records = sorted(records, key=lambda r: r["pair_id"])
    else:
        sorted_records = list(records)
    for r in sorted_records:
        groups_by_prompt.setdefault(r["prompt"], set()).add(r["group_id"])

    selections: list[HeldoutPrompt] = []
    for prompt, groups in groups_by_prompt.items():
        if rule == "first_alpha":
            group_id = sorted(groups)[0]
        elif rule == "first_in_record_order":
            for r in sorted_records:
                if r["prompt"] == prompt:
                    group_id = r["group_id"]
                    break
            else:  # pragma: no cover - groups is non-empty by construction
                raise RuntimeError(f"no record found for prompt: {prompt!r}")
        else:
            raise ValueError(f"unknown selection rule: {rule!r}")

        if group_id not in image_manifest:
            raise KeyError(
                f"canonical group {group_id} for prompt {_prompt_id(prompt)} "
                f"missing from t2/image_manifest.json"
            )
        entry = image_manifest[group_id]
        if entry.get("status") != "ok":
            raise ValueError(
                f"group {group_id}: T2 manifest status != 'ok' (got {entry.get('status')!r})"
            )
        cond_image_path = entry["image_path"]
        if not cond_image_path:
            raise ValueError(f"group {group_id}: empty image_path in T2 manifest")

        selections.append(
            HeldoutPrompt(
                prompt_id=_prompt_id(prompt),
                prompt=prompt,
                group_id=group_id,
                cond_image_path=cond_image_path,
            )
        )
    selections.sort(key=lambda s: s.prompt_id)
    if len(selections) != EXPECTED_HELDOUT_PROMPTS:
        raise RuntimeError(
            f"selected {len(selections)} canonical prompts, expected {EXPECTED_HELDOUT_PROMPTS}"
        )
    return selections


# ---------- generation_config + inference adapter ----------


def canonical_generation_config(
    seed: int,
    sampler: str = "unipc",
    inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = "",
    resolution: tuple[int, int] = (832, 480),
    num_frames: int = 81,
    dtype: str = "bfloat16",
    judge_preprocessing: str = "phygroundata.md#section-8",
) -> dict:
    """The single shared generation_config dict.

    Per AC-7.2: ``seed``, ``sampler``, ``inference_steps``,
    ``guidance_scale``, ``negative_prompt``, ``resolution``,
    ``num_frames=81``, ``dtype``, ``judge_preprocessing`` must be
    byte-identical between baseline and trained runs. The dict is
    serialized with ``sort_keys=True`` separators=(",", ":") so the
    bytes are stable across Python instances.

    Defaults mirror the parent plan / round-0 audit §5:
      - ``inference_steps`` = 50 (DiffSynth I2V default in
        ``inference_I2V_cont_after_train.py:71``)
      - ``num_frames`` = 81 (AC-3.5)
      - ``resolution`` = (832, 480) (parent-plan training resolution)
      - ``dtype`` = "bfloat16" (DEC-2 dtype policy)
    """
    return {
        "seed": int(seed),
        "sampler": str(sampler),
        "inference_steps": int(inference_steps),
        "guidance_scale": float(guidance_scale),
        "negative_prompt": str(negative_prompt),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "num_frames": int(num_frames),
        "dtype": str(dtype),
        "judge_preprocessing": str(judge_preprocessing),
    }


def serialize_generation_config(cfg: dict) -> bytes:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")


# Adapter callable: (run_kind, prompt, cond_image_path, gen_config_bytes,
#                    out_dir, ckpt_args) -> dict (sample manifest)
InferenceAdapter = Callable[[str, str, str, bytes, pathlib.Path, dict], dict]


def subprocess_inference_adapter(
    inference_smoke_py: pathlib.Path,
    python_executable: str = sys.executable,
    timeout_s: int = 60 * 60,
) -> InferenceAdapter:
    """Adapter that shells out to ``inference_smoke.py`` per sample.

    Contract assumed of ``inference_smoke.py`` (rl5, task #7):
      - CLI flags: ``--baseline_ckpt`` / ``--trained_lora`` / ``--low_noise_ckpt``
        (one of the first two depending on run_kind) plus ``--prompt``,
        ``--cond_image``, ``--out_dir``, ``--gen_config_json``, ``--seed``.
      - Reads gen_config_json bytes, asserts they're parseable, applies
        them as the generation_config, writes the resulting sample +
        per-run manifest JSON under out_dir.
      - Exits 0 on success, non-zero on failure (OOM/dtype/device).

    If rl5 lands a different CLI shape, swap this adapter for a Python
    API one (see ``python_api_inference_adapter`` below) without
    changing the orchestrator.
    """

    def adapter(
        run_kind: str,
        prompt: str,
        cond_image_path: str,
        gen_config_bytes: bytes,
        out_dir: pathlib.Path,
        ckpt_args: dict,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_cfg_path = out_dir / "gen_config.json"
        gen_cfg_path.write_bytes(gen_config_bytes)
        prompt_path = out_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        cmd = [
            python_executable,
            str(inference_smoke_py),
            "--prompt", prompt,
            "--cond_image", cond_image_path,
            "--out_dir", str(out_dir),
            "--gen_config_json", str(gen_cfg_path),
            "--seed", str(json.loads(gen_config_bytes)["seed"]),
        ]
        if run_kind == "baseline":
            if "baseline_ckpt" not in ckpt_args:
                raise KeyError("baseline run requires ckpt_args['baseline_ckpt']")
            cmd += ["--baseline_ckpt", ckpt_args["baseline_ckpt"]]
            if "low_noise_ckpt" in ckpt_args:
                cmd += ["--low_noise_ckpt", ckpt_args["low_noise_ckpt"]]
        elif run_kind == "trained":
            if "trained_lora" not in ckpt_args:
                raise KeyError("trained run requires ckpt_args['trained_lora']")
            cmd += ["--trained_lora", ckpt_args["trained_lora"]]
            if "baseline_ckpt" in ckpt_args:
                cmd += ["--baseline_ckpt", ckpt_args["baseline_ckpt"]]
            if "low_noise_ckpt" in ckpt_args:
                cmd += ["--low_noise_ckpt", ckpt_args["low_noise_ckpt"]]
        else:
            raise ValueError(f"unknown run_kind: {run_kind!r}")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        wall = time.time() - t0
        log_path = out_dir / "inference.log"
        log_path.write_text(
            f"# cmd: {' '.join(cmd)}\n# returncode: {proc.returncode}\n# wall_seconds: {wall:.2f}\n\n"
            f"-- stdout --\n{proc.stdout}\n\n-- stderr --\n{proc.stderr}",
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"inference_smoke.py failed for {run_kind} (returncode={proc.returncode}); "
                f"see {log_path}"
            )
        return {
            "run_kind": run_kind,
            "out_dir": str(out_dir),
            "wall_seconds": round(wall, 2),
            "cmd": cmd,
        }

    return adapter


def python_api_inference_adapter(run_one_sample: Callable) -> InferenceAdapter:
    """Adapter that calls into rl5's ``inference_smoke.run_one_sample`` directly.

    Expected signature (TBD — pending rl5 task #7 round-1 commit)::

        run_one_sample(
            run_kind: str,            # "baseline" | "trained"
            prompt: str,
            cond_image_path: str,
            gen_config: dict,
            out_dir: pathlib.Path,
            **ckpt_args,              # baseline_ckpt / trained_lora / low_noise_ckpt
        ) -> dict                     # per-run manifest

    Once rl5 lands the function, swap the adapter via the orchestrator's
    ``--adapter python_api`` flag (see ``main()``).
    """

    def adapter(
        run_kind: str,
        prompt: str,
        cond_image_path: str,
        gen_config_bytes: bytes,
        out_dir: pathlib.Path,
        ckpt_args: dict,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_cfg_path = out_dir / "gen_config.json"
        gen_cfg_path.write_bytes(gen_config_bytes)
        gen_config = json.loads(gen_config_bytes)
        t0 = time.time()
        result = run_one_sample(
            run_kind=run_kind,
            prompt=prompt,
            cond_image_path=cond_image_path,
            gen_config=gen_config,
            out_dir=out_dir,
            **ckpt_args,
        )
        wall = time.time() - t0
        return {
            "run_kind": run_kind,
            "out_dir": str(out_dir),
            "wall_seconds": round(wall, 2),
            "result": result,
        }

    return adapter


def dry_run_inference_adapter() -> InferenceAdapter:
    """No-op adapter for orchestration testing without GPU.

    Writes the gen_config + a stub ``sample.placeholder`` file so the
    full orchestration loop and per-prompt directory structure can be
    exercised without invoking the real inference pipeline.
    """

    def adapter(
        run_kind: str,
        prompt: str,
        cond_image_path: str,
        gen_config_bytes: bytes,
        out_dir: pathlib.Path,
        ckpt_args: dict,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gen_config.json").write_bytes(gen_config_bytes)
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (out_dir / "sample.placeholder").write_text(
            f"# dry-run: would have generated {run_kind} sample for prompt:\n# {prompt}\n",
            encoding="utf-8",
        )
        return {
            "run_kind": run_kind,
            "out_dir": str(out_dir),
            "wall_seconds": 0.0,
            "dry_run": True,
        }

    return adapter


# ---------- byte-identical generation_config check ----------


def assert_byte_identical_generation_configs(
    baseline_dir: pathlib.Path, trained_dir: pathlib.Path
) -> str:
    a = (baseline_dir / "gen_config.json").read_bytes()
    b = (trained_dir / "gen_config.json").read_bytes()
    if a != b:
        raise RuntimeError(
            f"generation_config drift between {baseline_dir} and {trained_dir}: "
            f"sha256={hashlib.sha256(a).hexdigest()[:16]} vs {hashlib.sha256(b).hexdigest()[:16]}"
        )
    return hashlib.sha256(a).hexdigest()


# ---------- per-prompt orchestration ----------


def regen_one_prompt(
    selection: HeldoutPrompt,
    gen_config_bytes: bytes,
    prompt_out_root: pathlib.Path,
    adapter: InferenceAdapter,
    ckpt_args: dict,
    resume: bool = True,
) -> dict:
    prompt_dir = prompt_out_root / selection.prompt_id
    prompt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = prompt_dir / "prompt_manifest.json"
    if resume and summary_path.exists():
        existing = json.loads(summary_path.read_bytes())
        if existing.get("complete") is True:
            existing["resumed"] = True
            return existing

    baseline_dir = prompt_dir / "baseline"
    trained_dir = prompt_dir / "trained"

    baseline_manifest = adapter(
        "baseline",
        selection.prompt,
        selection.cond_image_path,
        gen_config_bytes,
        baseline_dir,
        ckpt_args,
    )
    trained_manifest = adapter(
        "trained",
        selection.prompt,
        selection.cond_image_path,
        gen_config_bytes,
        trained_dir,
        ckpt_args,
    )

    cfg_sha = assert_byte_identical_generation_configs(baseline_dir, trained_dir)

    out = {
        "prompt_id": selection.prompt_id,
        "prompt": selection.prompt,
        "group_id": selection.group_id,
        "cond_image_path": selection.cond_image_path,
        "gen_config_sha256": cfg_sha,
        "baseline": baseline_manifest,
        "trained": trained_manifest,
        "complete": True,
    }
    summary_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out


def regen_all(
    selections: list[HeldoutPrompt],
    gen_config_bytes: bytes,
    out_root: pathlib.Path,
    adapter: InferenceAdapter,
    ckpt_args: dict,
    resume: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict]:
    """Drive the full 42-prompt loop. Optional rank/world_size shards
    selections across N processes by index modulo world_size — caller
    invokes one ``regen_all`` per rank.
    """
    prompt_out_root = out_root / "heldout_regen"
    prompt_out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for idx, sel in enumerate(selections):
        if idx % world_size != rank:
            continue
        rec = regen_one_prompt(sel, gen_config_bytes, prompt_out_root, adapter, ckpt_args, resume=resume)
        results.append(rec)
    return results


# ---------- CLI ----------


def main() -> int:
    p = argparse.ArgumentParser(description="M6 heldout regen orchestration scaffold")
    p.add_argument("--t0-t3-root", type=pathlib.Path, required=True,
                   help="<T0_T3_ROOT> as resolved by M0 audit (round-0 §1).")
    p.add_argument("--out-dir", type=pathlib.Path, required=True,
                   help="Run output root; per-prompt dirs land under <out_dir>/heldout_regen/<prompt_id>/.")
    p.add_argument("--recipes-dir", type=pathlib.Path, default=HERE / "recipes",
                   help="Path to the dpo_v0 recipes/ dir (defaults to alongside this file).")
    p.add_argument("--inference-smoke-py", type=pathlib.Path, default=HERE / "inference_smoke.py",
                   help="Path to rl5's inference_smoke.py (used by subprocess adapter).")
    p.add_argument("--adapter", choices=["subprocess", "dry_run"], default="subprocess",
                   help="Inference adapter to use. 'subprocess' shells out to inference_smoke.py; "
                        "'dry_run' is a no-op for orchestration testing without GPU.")
    p.add_argument("--baseline-ckpt", type=str, default=None,
                   help="Path to original-init no-DPO baseline ckpt (high-noise sharded base).")
    p.add_argument("--trained-lora", type=str, default=None,
                   help="Path to LoRA adapter checkpoint from M3/M4 trained run.")
    p.add_argument("--low-noise-ckpt", type=str, default=None,
                   help="Path to frozen low-noise expert (per AC-7.3 SHA stamping).")
    p.add_argument("--seed", type=int, default=42,
                   help="Generation seed (byte-identical between baseline and trained).")
    p.add_argument("--sampler", type=str, default="unipc")
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--resolution", type=str, default="832x480",
                   help="WxH; defaults to 832x480 per parent-plan training resolution.")
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--selection-rule", choices=["first_alpha", "first_in_record_order"],
                   default="first_alpha")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable per-prompt skip-if-complete idempotency.")
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--compute-envelope", choices=["single_gpu", "dpo_multi_gpu_zero2"],
                   default="single_gpu",
                   help="DEC-6 envelope tag stamped into the run manifest.")
    args = p.parse_args()

    if args.adapter == "subprocess":
        if args.baseline_ckpt is None or args.trained_lora is None:
            print(
                "[heldout_regen] --baseline-ckpt and --trained-lora are required with --adapter subprocess",
                file=sys.stderr,
            )
            return 2

    # 1. recipe pin
    recipe_id = assert_recipe_pin(args.recipes_dir, EXPECTED_RECIPE_ID)
    print(f"[recipe] pin OK: {recipe_id}", flush=True)

    # 2. heldout split
    records = load_heldout_records(args.t0_t3_root)
    image_manifest = load_t2_image_manifest(args.t0_t3_root)
    print(
        f"[heldout] {len(records)} pairs / {len({r['group_id'] for r in records})} groups / "
        f"{len({r['prompt'] for r in records})} unique prompts loaded from {args.t0_t3_root}",
        flush=True,
    )

    # 3. canonical selection
    selections = select_canonical_groups(records, image_manifest, rule=args.selection_rule)
    print(f"[selection] {len(selections)} canonical (prompt, group, image) triples", flush=True)

    # 4. byte-identical generation_config
    res = args.resolution.lower().split("x")
    if len(res) != 2:
        raise SystemExit(f"--resolution must be WxH, got {args.resolution!r}")
    gen_config = canonical_generation_config(
        seed=args.seed,
        sampler=args.sampler,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        resolution=(int(res[0]), int(res[1])),
        num_frames=args.num_frames,
        dtype=args.dtype,
    )
    gen_config_bytes = serialize_generation_config(gen_config)
    cfg_sha = hashlib.sha256(gen_config_bytes).hexdigest()
    print(f"[gen_config] sha256={cfg_sha[:16]} ({len(gen_config_bytes)} bytes, sort_keys=True)", flush=True)

    # 5. adapter
    if args.adapter == "subprocess":
        if not args.inference_smoke_py.exists():
            print(
                f"[heldout_regen] inference_smoke.py not found at {args.inference_smoke_py}; "
                f"either pass --inference-smoke-py <path> or wait for rl5 task #7 to land.",
                file=sys.stderr,
            )
            return 2
        adapter = subprocess_inference_adapter(args.inference_smoke_py)
    else:
        adapter = dry_run_inference_adapter()

    ckpt_args = {}
    if args.baseline_ckpt is not None:
        ckpt_args["baseline_ckpt"] = args.baseline_ckpt
    if args.trained_lora is not None:
        ckpt_args["trained_lora"] = args.trained_lora
    if args.low_noise_ckpt is not None:
        ckpt_args["low_noise_ckpt"] = args.low_noise_ckpt

    # 6. run-level dir + manifest
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_pre = {
        "ts_utc": ts,
        "t0_t3_root": str(args.t0_t3_root),
        "recipe_id": recipe_id,
        "compute_envelope": args.compute_envelope,
        "selection_rule": args.selection_rule,
        "n_selections": len(selections),
        "gen_config": gen_config,
        "gen_config_sha256": cfg_sha,
        "ckpt_args": ckpt_args,
        "rank": args.rank,
        "world_size": args.world_size,
    }
    (run_dir / "run_manifest.pre.json").write_text(
        json.dumps(run_manifest_pre, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 7. orchestration loop
    results = regen_all(
        selections=selections,
        gen_config_bytes=gen_config_bytes,
        out_root=run_dir,
        adapter=adapter,
        ckpt_args=ckpt_args,
        resume=not args.no_resume,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_manifest_post = dict(run_manifest_pre)
    run_manifest_post["results_count"] = len(results)
    run_manifest_post["complete"] = all(r.get("complete") for r in results)
    run_manifest_post["results"] = results
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest_post, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[done] regenerated {len(results)} prompts (rank {args.rank}/{args.world_size}); "
        f"manifest at {run_dir / 'run_manifest.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
