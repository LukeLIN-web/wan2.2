"""M6 heldout regeneration orchestrator.

Reads ``<T0_T3_ROOT>/splits/heldout.json`` (579 pairs / 245 groups / 42
unique prompts) and ``<T0_T3_ROOT>/t2/image_manifest.json`` (1440 group
entries), picks one canonical (group, image) per prompt under a
deterministic rule, and runs ``inference_smoke.py`` twice per prompt
(baseline ckpt + trained ckpt) under one byte-identical generation
config.

The actual generation work is delegated to ``inference_smoke.py`` —
this module calls it via subprocess CLI (preferred for byte-identical
config + a clean process boundary) or via the in-process Python API
``inference_smoke.run_one_sample(...)``.

Plan: humanize/i2v.md AC-7.1, AC-7.2, AC-7.3, M6.

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
from typing import Any, Callable, Optional


HERE = pathlib.Path(__file__).resolve().parent
DPO_ROOT = HERE.parent  # humanize/dpo_v0/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DPO_ROOT))

# rl8 task #16 manifest_writer (commit e096edf, schema version 1).
# Adopted: assert_recipe_pins (3-way pin assert returning full pins dict)
# + atomic_write_json (deterministic key order, .tmp + rename).
from dataprocessing.manifest_writer import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION as _MANIFEST_WRITER_SCHEMA_VERSION,
    assert_recipe_pins as _mw_assert_recipe_pins,
    atomic_write_json as _mw_atomic_write_json,
)

EXPECTED_RECIPE_ID = "6bef6e104cdd3442"  # AC-3.1, frozen in task-6 round 1
EXPECTED_HELDOUT_PROMPTS = 42  # round-0 audit §3
EXPECTED_HELDOUT_GROUPS = 245
EXPECTED_HELDOUT_PAIRS = 579


# ---------- canonical artifact loaders ----------


def assert_recipe_pin(recipes_dir: pathlib.Path, expected: str = EXPECTED_RECIPE_ID) -> str:
    """Thin wrapper around manifest_writer.assert_recipe_pins (3-way assert).

    Returns the recipe_id string for backward compatibility with the
    earlier 2-way wrapper. Callers wanting the full pins dict (for
    splatting into a manifest) should call ``_mw_assert_recipe_pins``
    directly.
    """
    pins = _mw_assert_recipe_pins(recipes_dir, expected_recipe_id=expected)
    return pins["recipe_id"]


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


def _resolve_cond_image_path(
    primary: str,
    fallback_root: Optional[pathlib.Path],
) -> str:
    """Resolve a canonical T2 image path, with optional cross-machine fallback.

    Mirrors the trainer's ``--cond-image-fallback-root`` semantics from
    rl1 commit ``0030832`` (round-3 attempt 5): if the primary absolute
    path exists, return it; otherwise look for ``<fallback_root>/<basename>``
    and return that if present. The fallback is basename-keyed because
    the canonical manifest paths come from the dev box and may live
    under different prefixes on juyi-finetune / juyi-videorl.

    Returns the resolved path string. If neither primary nor fallback
    exists, returns the primary path unchanged so the downstream
    inference_smoke loader fails loudly with the real error rather than
    a synthesized one (caller can always pre-check existence if needed).
    """
    if pathlib.Path(primary).exists():
        return primary
    if fallback_root is None:
        return primary
    candidate = fallback_root / pathlib.Path(primary).name
    if candidate.exists():
        return str(candidate)
    return primary


def select_canonical_groups(
    records: list[dict],
    image_manifest: dict,
    rule: str = "first_alpha",
    cond_image_fallback_root: Optional[pathlib.Path] = None,
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
        cond_image_path = _resolve_cond_image_path(cond_image_path, cond_image_fallback_root)

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
    sampler: str = "uni_pc",
    inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = "",
    resolution: tuple[int, int] = (832, 480),
    num_frames: int = 81,
    dtype: str = "bf16",
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


# Filenames inference_smoke.py is expected to drop in its out_dir
# (per rl5 schema in #dpo:5db9718b msg `a76f4c8e`). The `ckpt_shas`
# block lives at the top level of either file; we look at both.
_INFERENCE_SMOKE_MANIFEST_NAMES = ("manifest.json", "run_manifest.json")


def _extract_ckpt_shas(out_dir: pathlib.Path) -> Optional[dict]:
    """Read inference_smoke.py's per-run manifest and surface ``ckpt_shas``.

    Returns ``None`` if no manifest exists yet (e.g. dry-run, or the
    inference smoke step did not write one). Caller decides whether the
    absence is a hard fail (M6 production) or expected (dry_run / smoke
    pre-launch). Per AC-7.3, ``ckpt_shas`` is the authoritative
    "what was actually loaded" record (rl5 `a76f4c8e` option B).
    """
    for name in _INFERENCE_SMOKE_MANIFEST_NAMES:
        path = out_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_bytes())
        except json.JSONDecodeError:
            continue
        shas = data.get("ckpt_shas")
        if isinstance(shas, dict):
            return shas
    return None


def subprocess_inference_adapter(
    inference_smoke_py: pathlib.Path,
    python_executable: str = sys.executable,
    timeout_s: int = 60 * 60,
) -> InferenceAdapter:
    """Adapter that shells out to ``inference_smoke.py`` per sample.

    CLI contract verified against rl5's actual ``inference_smoke.py`` on
    ``rlcr/task-6`` (round-2 commit ``ac00949``, cherry-picked as
    ``6b27fe7``). Earlier spec at msg ``5e8993eb`` drifted from the
    landed implementation — this adapter targets the IMPLEMENTED CLI
    (rl1 caught the drift in #dpo:dac89b67 msg ``04170e0b``):

        inference_smoke.py
          --upstream PATH                       # canonical Wan2.2-I2V-A14B root
          --mode {both, baseline, trained}     # M6 uses baseline | trained
          --lora-adapter PATH                  # trained mode only; baseline rejects it
          --low-noise-ckpt PATH                # optional override; defaults to <upstream>/low_noise_model/
          --gen-config-json PATH               # single source of truth bytes
          --recipe-yaml PATH                   # optional; defaults to recipes/wan22_i2v_a14b__round2_v0.yaml
          --prompt TEXT
          --cond-image PATH
          --out-dir PATH
          --seed INT                           # default 0
          --compute-envelope {single_gpu, dpo_multi_gpu_ddp,
                              dpo_multi_gpu_zero2,
                              multi_gpu_inference_seed_parallel}

    M6 invokes one prompt at a time, ``--mode baseline`` then
    ``--mode trained``, with the same ``--gen-config-json`` so
    byte-identicality is enforced by the file itself.

    ``ckpt_args`` keys (orchestrator-level, translated to rl5 CLI flags):
      * ``upstream`` (REQUIRED) — canonical I2V-A14B root.
      * ``trained_lora`` (REQUIRED for run_kind="trained") — LoRA
        safetensors path; passed as ``--lora-adapter``. Baseline takes NO
        adapter (inference_smoke rejects it at argparse).
      * ``low_noise_ckpt`` (optional) — explicit low-noise override.
      * ``recipe_yaml`` (optional) — canonical recipe YAML override.
      * ``compute_envelope`` (optional) — per-run envelope tag.
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

        if "upstream" not in ckpt_args:
            raise KeyError(
                "inference_smoke.py requires ckpt_args['upstream'] "
                "(canonical Wan2.2-I2V-A14B root)."
            )

        cmd = [
            python_executable,
            str(inference_smoke_py),
            "--upstream", str(ckpt_args["upstream"]),
            "--mode", run_kind,
            "--prompt", prompt,
            "--cond-image", cond_image_path,
            "--out-dir", str(out_dir),
            "--gen-config-json", str(gen_cfg_path),
            "--seed", str(json.loads(gen_config_bytes)["seed"]),
        ]
        if "low_noise_ckpt" in ckpt_args:
            cmd += ["--low-noise-ckpt", str(ckpt_args["low_noise_ckpt"])]
        if "recipe_yaml" in ckpt_args:
            cmd += ["--recipe-yaml", str(ckpt_args["recipe_yaml"])]
        if "compute_envelope" in ckpt_args:
            cmd += ["--compute-envelope", str(ckpt_args["compute_envelope"])]

        if run_kind == "baseline":
            # Baseline takes NO --lora-adapter (inference_smoke rejects it).
            pass
        elif run_kind == "trained":
            if "trained_lora" not in ckpt_args:
                raise KeyError(
                    "trained run requires ckpt_args['trained_lora'] "
                    "(LoRA safetensors path; passed as --lora-adapter)."
                )
            cmd += ["--lora-adapter", str(ckpt_args["trained_lora"])]
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
            "ckpt_shas": _extract_ckpt_shas(out_dir),
        }

    return adapter


def python_api_inference_adapter(
    run_one_sample: Callable,
    UpstreamPaths: Optional[Callable] = None,
    assert_recipe_pin: Optional[Callable] = None,
    cache_pipe: bool = False,
    build_pipeline: Optional[Callable] = None,
) -> InferenceAdapter:
    """Adapter that calls into rl5's ``inference_smoke.run_one_sample`` directly.

    ACTUAL signature (verified against ``rlcr/task-6`` HEAD; rl1 caught
    the spec/impl drift in #dpo:dac89b67 msg ``04170e0b``)::

        run_one_sample(
            *,
            mode,                    # "baseline" | "trained"
            out_dir: Path,
            upstream: UpstreamPaths, # dataclass(upstream_root=Path)
            torch_dtype,             # torch.dtype, e.g. torch.bfloat16
            device: str,             # "cuda" | "cpu"
            prompt: str,
            cond_image_path: Path,
            generation_config: dict,
            generation_config_bytes: bytes,
            lora_adapter_path: Optional[Path],   # None for baseline
            compute_envelope: str,
            recipe_id: str,                      # the 16-char pin, NOT the yaml path
            high_noise_sha: Optional[str] = None,
            low_noise_sha: Optional[str] = None,
            pipe: Any = None,                    # cache_pipe path
            lora_already_attached: bool = False, # cache_pipe path
            lora_sha: Optional[str] = None,      # cache_pipe path
        ) -> dict[str, Any]            # returns manifest dict incl. ckpt_shas

    Pass the matching ``UpstreamPaths`` dataclass as the second arg
    (typically ``inference_smoke.UpstreamPaths``); the adapter
    constructs ``UpstreamPaths(upstream_root=Path(ckpt_args["upstream"]))``
    each call. The recipe_id is read via ``assert_recipe_pin`` (default
    is the orchestrator's local helper which delegates to manifest_writer).

    Subprocess adapter remains the operational default — Python API
    is opt-in. The process boundary is an escape valve for deploy
    issues (env divergence, partial OOM recovery, GPU-context leak)
    where the in-process adapter would take down the orchestrator.

    ``cache_pipe`` (False by default — backward compat): when True, the
    adapter holds a single ``WanVideoPipeline`` across all calls,
    skipping the ~95 GB ``build_pipeline`` reload that dominates wall
    time when one process serves N>>1 prompts. Required for full
    42-prompt heldout regen at modest world_size (e.g. world_size=8 →
    10-11 calls/rank → 10-11×95 GB rebuilds without this flag).

    With ``cache_pipe=True`` the caller MUST iterate mode-batched per
    rank (all baselines first, then all trained) — DiffSynth does not
    expose a clean LoRA detach path, so a baseline call after a trained
    call requires a pipe rebuild. The orchestrator's ``--mode-batched``
    flag enforces this.

    ``build_pipeline`` overrides the default (lazy-imported from
    ``inference_smoke``); pass an explicit callable in tests so the
    cache path is exercisable without GPU / DiffSynth.
    """
    if assert_recipe_pin is None:
        # Late-bound to module-level wrapper to avoid forward-reference at
        # import time. The module-level helper itself shells through to
        # manifest_writer.assert_recipe_pins.
        _assert_recipe_pin = globals()["assert_recipe_pin"]
    else:
        _assert_recipe_pin = assert_recipe_pin

    # Process-local cache of the built pipeline + LoRA attachment + sharded
    # SHA hashes. Mutated in-place across calls when cache_pipe=True; ignored
    # otherwise. Keyed by (upstream_root, torch_dtype_name, device) — any
    # mismatch across calls is a caller error and raises.
    _state: dict[str, Any] = {
        "pipe": None,
        "lora_attached_path": None,
        "upstream_root": None,
        "torch_dtype_name": None,
        "device": None,
        "lora_sha": None,
        "high_noise_sha": None,
        "low_noise_sha": None,
    }

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
        generation_config = json.loads(gen_config_bytes)

        if "upstream" not in ckpt_args:
            raise KeyError(
                "python_api adapter requires ckpt_args['upstream'] "
                "(canonical Wan2.2-I2V-A14B root)."
            )
        upstream_root = pathlib.Path(ckpt_args["upstream"])

        if UpstreamPaths is None:
            try:
                from inference_smoke import UpstreamPaths as _UpstreamPaths  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "python_api adapter needs the UpstreamPaths dataclass either passed "
                    "explicitly or importable from inference_smoke; got: "
                    f"{exc!r}"
                ) from exc
            upstream = _UpstreamPaths(upstream_root=upstream_root)
        else:
            upstream = UpstreamPaths(upstream_root=upstream_root)

        # Resolve recipe_id (16-char pin string) from recipe_yaml override
        # or default canonical layout under <orchestrator-pkg>/recipes/.
        recipes_dir = (
            pathlib.Path(ckpt_args["recipe_yaml"]).parent
            if "recipe_yaml" in ckpt_args
            else DPO_ROOT / "recipes"
        )
        recipe_id = _assert_recipe_pin(recipes_dir)

        # torch_dtype + device: import lazily so the orchestrator's CPU-only
        # paths (subprocess + dry_run) don't need torch loaded.
        import torch  # type: ignore
        dtype_name = generation_config.get("dtype", "bf16")
        torch_dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)
        device = ckpt_args.get("device", "cuda")

        lora_adapter_path: Optional[pathlib.Path]
        if run_kind == "baseline":
            lora_adapter_path = None
        elif run_kind == "trained":
            if "trained_lora" not in ckpt_args:
                raise KeyError(
                    "trained run requires ckpt_args['trained_lora'] "
                    "(LoRA safetensors path)."
                )
            lora_adapter_path = pathlib.Path(ckpt_args["trained_lora"])
        else:
            raise ValueError(f"unknown run_kind: {run_kind!r}")

        compute_envelope = ckpt_args.get("compute_envelope", "single_gpu")

        # Cache path: pre-build the pipe on first call, validate state
        # consistency on subsequent calls, and decide whether to skip the
        # LoRA attach inside run_one_sample.
        pipe_kwargs: dict[str, Any] = {}
        if cache_pipe:
            # Validate (upstream_root, torch_dtype, device) stable across
            # calls; mismatch signals an orchestrator error.
            if _state["pipe"] is not None:
                if _state["upstream_root"] != upstream_root:
                    raise RuntimeError(
                        "cached pipe was built for upstream_root "
                        f"{_state['upstream_root']}; got {upstream_root} "
                        "on later call. Rebuild required."
                    )
                if _state["torch_dtype_name"] != dtype_name:
                    raise RuntimeError(
                        f"cached pipe torch_dtype was {_state['torch_dtype_name']}; "
                        f"got {dtype_name} on later call. Rebuild required."
                    )
                if _state["device"] != device:
                    raise RuntimeError(
                        f"cached pipe device was {_state['device']}; "
                        f"got {device} on later call. Rebuild required."
                    )
            else:
                # First call: build pipe up-front so the adapter (not
                # run_one_sample) owns the cached reference.
                if build_pipeline is None:
                    try:
                        from inference_smoke import build_pipeline as _build_pipeline  # type: ignore
                    except ImportError as exc:
                        raise ImportError(
                            "cache_pipe=True requires inference_smoke.build_pipeline "
                            "importable from PYTHONPATH; got: " f"{exc!r}"
                        ) from exc
                else:
                    _build_pipeline = build_pipeline
                print(
                    f"[python_api_adapter] cache_pipe=True: building shared pipeline "
                    f"once (upstream={upstream_root}, dtype={dtype_name}, device={device})",
                    flush=True,
                )
                _state["pipe"] = _build_pipeline(
                    upstream, torch_dtype=torch_dtype, device=device
                )
                _state["upstream_root"] = upstream_root
                _state["torch_dtype_name"] = dtype_name
                _state["device"] = device

            # LoRA state management (no detach path in DiffSynth → caller
            # MUST iterate mode-batched: all baselines, then all trained).
            if run_kind == "baseline":
                if _state["lora_attached_path"] is not None:
                    raise RuntimeError(
                        "cached pipe has LoRA "
                        f"{_state['lora_attached_path']} attached; baseline "
                        "run after trained run requires pipe rebuild because "
                        "DiffSynth does not expose a clean detach. Iterate "
                        "mode-batched (all baselines first, then all trained) "
                        "via the orchestrator's --mode-batched flag."
                    )
                lora_already_attached = False
            else:  # trained
                attached = _state["lora_attached_path"]
                if attached is not None and attached != lora_adapter_path:
                    raise RuntimeError(
                        f"cached pipe has LoRA {attached} attached; trained "
                        f"run requested {lora_adapter_path}. Different LoRA "
                        "paths within the same cached session require pipe "
                        "rebuild (no detach support)."
                    )
                lora_already_attached = (attached == lora_adapter_path)

            pipe_kwargs["pipe"] = _state["pipe"]
            pipe_kwargs["lora_already_attached"] = lora_already_attached
            if lora_already_attached and _state.get("lora_sha"):
                pipe_kwargs["lora_sha"] = _state["lora_sha"]
            if _state.get("high_noise_sha"):
                pipe_kwargs["high_noise_sha"] = _state["high_noise_sha"]
            if _state.get("low_noise_sha"):
                pipe_kwargs["low_noise_sha"] = _state["low_noise_sha"]

        t0 = time.time()
        result = run_one_sample(
            mode=run_kind,
            out_dir=out_dir,
            upstream=upstream,
            torch_dtype=torch_dtype,
            device=device,
            prompt=prompt,
            cond_image_path=pathlib.Path(cond_image_path),
            generation_config=generation_config,
            generation_config_bytes=gen_config_bytes,
            lora_adapter_path=lora_adapter_path,
            compute_envelope=compute_envelope,
            recipe_id=recipe_id,
            **pipe_kwargs,
        )
        wall = time.time() - t0

        # Update cached state from manifest shas + lora attach side-effects.
        if cache_pipe:
            shas = result.get("ckpt_shas") if isinstance(result, dict) else None
            if isinstance(shas, dict):
                if shas.get("high_noise_base_sha256"):
                    _state["high_noise_sha"] = shas["high_noise_base_sha256"]
                if shas.get("low_noise_frozen_sha256"):
                    _state["low_noise_sha"] = shas["low_noise_frozen_sha256"]
                if run_kind == "trained" and shas.get("lora_adapter_sha256"):
                    _state["lora_sha"] = shas["lora_adapter_sha256"]
            if run_kind == "trained":
                _state["lora_attached_path"] = lora_adapter_path

        return {
            "run_kind": run_kind,
            "out_dir": str(out_dir),
            "wall_seconds": round(wall, 2),
            "result": result,
            "ckpt_shas": (result.get("ckpt_shas") if isinstance(result, dict) else None)
                         or _extract_ckpt_shas(out_dir),
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
    a_sha = hashlib.sha256(a).hexdigest()
    if a != b:
        raise RuntimeError(
            f"generation_config drift between {baseline_dir} and {trained_dir}: "
            f"sha256={a_sha[:16]} vs {hashlib.sha256(b).hexdigest()[:16]}"
        )
    return a_sha


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
    _mw_atomic_write_json(summary_path, out)
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
    mode_batched: bool = False,
) -> list[dict]:
    """Drive the full 42-prompt loop. Optional rank/world_size shards
    selections across N processes by index modulo world_size — caller
    invokes one ``regen_all`` per rank.

    ``mode_batched`` (False by default): when True, iterate each rank's
    assigned prompts in two passes — all baselines first, then all
    trained — instead of interleaving baseline+trained per prompt. This
    is required when the adapter caches a single pipeline across calls
    (``python_api_inference_adapter(..., cache_pipe=True)``) because
    DiffSynth does not expose a clean LoRA detach path: a trained call
    leaves LoRA attached on ``pipe.dit``, so the next baseline call on
    the same pipe would silently include the trained-mode weights.
    Mode-batched ordering means LoRA gets attached exactly once per rank
    (between the baseline pass and the trained pass) and stays attached
    for the entire trained pass.

    Resume semantics under ``mode_batched``: the per-prompt summary
    (``prompt_manifest.json``) is written only when both modes complete,
    matching the non-batched path; partial completion leaves
    per-mode dirs in place and the next run's resume check rebuilds
    state from those dirs (the inner ``regen_one_prompt`` skip-if-complete
    short-circuit kicks in for any prompt whose summary already exists).
    """
    prompt_out_root = out_root / "heldout_regen"
    prompt_out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    rank_selections = [
        sel for idx, sel in enumerate(selections) if idx % world_size == rank
    ]

    if not mode_batched:
        for sel in rank_selections:
            rec = regen_one_prompt(
                sel, gen_config_bytes, prompt_out_root, adapter, ckpt_args, resume=resume
            )
            results.append(rec)
        return results

    # Mode-batched path: pass 1 = all baselines, pass 2 = all trained,
    # then write per-prompt summary by zipping baseline + trained results.
    print(
        f"[regen_all] mode_batched=True: rank={rank}/{world_size} "
        f"baseline-pass over {len(rank_selections)} prompts",
        flush=True,
    )
    baseline_manifests: dict[str, dict] = {}
    for sel in rank_selections:
        prompt_dir = prompt_out_root / sel.prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        summary_path = prompt_dir / "prompt_manifest.json"
        if resume and summary_path.exists():
            existing = json.loads(summary_path.read_bytes())
            if existing.get("complete") is True:
                existing["resumed"] = True
                results.append(existing)
                continue
        baseline_dir = prompt_dir / "baseline"
        baseline_manifests[sel.prompt_id] = adapter(
            "baseline",
            sel.prompt,
            sel.cond_image_path,
            gen_config_bytes,
            baseline_dir,
            ckpt_args,
        )

    print(
        f"[regen_all] mode_batched=True: rank={rank}/{world_size} "
        f"trained-pass over {len(baseline_manifests)} prompts (baseline pass done)",
        flush=True,
    )
    for sel in rank_selections:
        if sel.prompt_id not in baseline_manifests:
            # Already resumed (full prompt summary existed); skip both passes.
            continue
        prompt_dir = prompt_out_root / sel.prompt_id
        baseline_dir = prompt_dir / "baseline"
        trained_dir = prompt_dir / "trained"
        trained_manifest = adapter(
            "trained",
            sel.prompt,
            sel.cond_image_path,
            gen_config_bytes,
            trained_dir,
            ckpt_args,
        )
        cfg_sha = assert_byte_identical_generation_configs(baseline_dir, trained_dir)
        out = {
            "prompt_id": sel.prompt_id,
            "prompt": sel.prompt,
            "group_id": sel.group_id,
            "cond_image_path": sel.cond_image_path,
            "gen_config_sha256": cfg_sha,
            "baseline": baseline_manifests[sel.prompt_id],
            "trained": trained_manifest,
            "complete": True,
        }
        _mw_atomic_write_json(prompt_dir / "prompt_manifest.json", out)
        results.append(out)
    return results


# ---------- CLI ----------


def main() -> int:
    p = argparse.ArgumentParser(description="M6 heldout regen orchestration scaffold")
    p.add_argument("--t0-t3-root", type=pathlib.Path, required=True,
                   help="<T0_T3_ROOT> as resolved by M0 audit (round-0 §1).")
    p.add_argument("--out-dir", type=pathlib.Path, required=True,
                   help="Run output root; per-prompt dirs land under <out_dir>/heldout_regen/<prompt_id>/.")
    p.add_argument("--recipes-dir", type=pathlib.Path, default=DPO_ROOT / "recipes",
                   help="Path to the dpo_v0 recipes/ dir (defaults to alongside this file).")
    p.add_argument("--inference-smoke-py", type=pathlib.Path, default=HERE / "inference_smoke.py",
                   help="Path to rl5's inference_smoke.py (used by subprocess adapter).")
    p.add_argument("--adapter", choices=["subprocess", "python_api", "dry_run"], default="subprocess",
                   help="Inference adapter to use. 'subprocess' shells out to inference_smoke.py "
                        "(default; deploy escape valve via process boundary). 'python_api' imports "
                        "inference_smoke.run_one_sample and calls it in-process (rl5 lock 5e8993eb / "
                        "ac00949). 'dry_run' is a no-op for orchestration testing without GPU.")
    p.add_argument("--trained-lora", type=str, default=None,
                   help="Path to LoRA adapter safetensors from M3/M4. REQUIRED with "
                        "--adapter subprocess|python_api. Passed as --lora-adapter to "
                        "inference_smoke for the trained run; baseline run takes no adapter.")
    p.add_argument("--low-noise-ckpt", type=str, default=None,
                   help="Optional explicit path to frozen low-noise expert; if omitted, "
                        "inference_smoke defaults to <upstream>/low_noise_model/ "
                        "(canonical sharded layout, AC-7.3 SHA stamped either way).")
    p.add_argument("--upstream", type=str, default=None,
                   help="Canonical Wan2.2-I2V-A14B root. REQUIRED with --adapter subprocess|"
                        "python_api. Used to construct UpstreamPaths for the python_api adapter "
                        "and passed as --upstream to inference_smoke's CLI.")
    p.add_argument("--recipe-yaml", type=str, default=None,
                   help="Optional recipe YAML override. Default = orchestrator-local "
                        "humanize/dpo_v0/recipes/wan22_i2v_a14b__round2_v0.yaml. "
                        "Passed as --recipe-yaml to inference_smoke.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for python_api adapter (passed to run_one_sample). "
                        "Subprocess adapter relies on inference_smoke's --device.")
    p.add_argument("--seed", type=int, default=42,
                   help="Generation seed (byte-identical between baseline and trained).")
    p.add_argument("--sampler", type=str, default="uni_pc",
                   help="Sampler name; defaults to 'uni_pc' per rl5 inference_smoke.py CLI lock.")
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--resolution", type=str, default="832x480",
                   help="WxH; defaults to 832x480 per parent-plan training resolution.")
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--dtype", type=str, default="bf16",
                   help="dtype tag stamped into gen_config; defaults to 'bf16' per rl5 schema.")
    p.add_argument("--selection-rule", choices=["first_alpha", "first_in_record_order"],
                   default="first_alpha")
    p.add_argument("--cond-image-fallback-root", type=pathlib.Path, default=None,
                   help="Basename-keyed fallback dir for cond images when canonical T2 paths "
                        "are not mounted on this host. If primary path missing, look for "
                        "<fallback_root>/<basename>; otherwise leave path as-is so loader "
                        "fails with the real error.")
    p.add_argument("--limit-prompts", type=int, default=None,
                   help="If set, run only the first N prompts after canonical selection "
                        "(deterministic order by prompt_id). Used for M6 partial-heldout scope "
                        "(e.g. 8-prompt smoke before full 42-prompt run). Applied AFTER "
                        "selection so the same N prompts always come out, regardless of rank "
                        "sharding (sharding is index-mod-world_size on top).")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable per-prompt skip-if-complete idempotency.")
    p.add_argument("--cache-pipe", action="store_true",
                   help="python_api adapter only: build the WanVideoPipeline (~95 GB) "
                        "ONCE per rank-process and reuse it across all assigned prompts. "
                        "Without this, each (prompt, mode) call rebuilds the pipeline — "
                        "fine for round-2 M6 (16 ranks × 1 call/rank) but blows up for "
                        "full 42-prompt × world_size=8 (10-11 calls/rank). Implies "
                        "--mode-batched (DiffSynth has no LoRA detach path; trained run "
                        "after baseline run requires LoRA attached, then it cannot be "
                        "removed without rebuilding pipe).")
    p.add_argument("--mode-batched", action="store_true",
                   help="Per rank, run all baseline prompts first, then all trained "
                        "prompts. Required when --cache-pipe is set; safe to enable "
                        "without it (only changes call ordering, not byte-equality). "
                        "Mode-batched ordering means LoRA attaches exactly once per rank "
                        "(between baseline pass and trained pass).")
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--compute-envelope",
                   choices=["single_gpu", "dpo_multi_gpu_ddp", "dpo_multi_gpu_zero2",
                            "multi_gpu_inference_seed_parallel"],
                   default="single_gpu",
                   help="DEC-6 envelope tag stamped into the run manifest. "
                        "Inference defaults to 'single_gpu'; M6 4-rank parallel-by-seed is "
                        "'multi_gpu_inference_seed_parallel' (DEC-6 i2v.md L76).")
    args = p.parse_args()

    if args.adapter in ("subprocess", "python_api"):
        missing = []
        if args.upstream is None:
            missing.append("--upstream")
        if args.trained_lora is None:
            missing.append("--trained-lora")
        if missing:
            print(
                f"[heldout_regen] {', '.join(missing)} required with --adapter "
                f"{args.adapter} (per rl5 inference_smoke.py actual CLI / Python API).",
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
    selections = select_canonical_groups(
        records,
        image_manifest,
        rule=args.selection_rule,
        cond_image_fallback_root=args.cond_image_fallback_root,
    )
    print(f"[selection] {len(selections)} canonical (prompt, group, image) triples", flush=True)
    if args.cond_image_fallback_root is not None:
        n_fallback = sum(
            1 for s in selections
            if str(args.cond_image_fallback_root) in s.cond_image_path
        )
        print(f"[selection] {n_fallback}/{len(selections)} cond images resolved via fallback", flush=True)

    # 3b. partial-heldout slice (M6 partial scope before full 42-prompt run)
    if args.limit_prompts is not None:
        if args.limit_prompts < 1:
            raise SystemExit(f"--limit-prompts must be >= 1, got {args.limit_prompts}")
        if args.limit_prompts > len(selections):
            raise SystemExit(
                f"--limit-prompts {args.limit_prompts} > {len(selections)} canonical prompts available"
            )
        selections = selections[: args.limit_prompts]
        print(
            f"[selection] sliced to first {args.limit_prompts} prompts (M6 partial scope); "
            f"deterministic order by prompt_id",
            flush=True,
        )

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
                f"pass --inference-smoke-py <path>.",
                file=sys.stderr,
            )
            return 2
        adapter = subprocess_inference_adapter(args.inference_smoke_py)
    elif args.adapter == "python_api":
        # Lazy-import so subprocess + dry_run paths don't require inference_smoke
        # to be importable (it depends on heavy GPU libs that the orchestrator
        # itself doesn't need).
        try:
            from inference_smoke import run_one_sample  # type: ignore
        except ImportError as exc:
            print(
                f"[heldout_regen] --adapter python_api requires inference_smoke.run_one_sample "
                f"importable from PYTHONPATH. ImportError: {exc}",
                file=sys.stderr,
            )
            return 2
        adapter = python_api_inference_adapter(run_one_sample, cache_pipe=args.cache_pipe)
    else:
        adapter = dry_run_inference_adapter()

    ckpt_args = {}
    if args.trained_lora is not None:
        ckpt_args["trained_lora"] = args.trained_lora
    if args.low_noise_ckpt is not None:
        ckpt_args["low_noise_ckpt"] = args.low_noise_ckpt
    if args.upstream is not None:
        ckpt_args["upstream"] = args.upstream
    if args.recipe_yaml is not None:
        ckpt_args["recipe_yaml"] = args.recipe_yaml
    if args.compute_envelope is not None:
        ckpt_args["compute_envelope"] = args.compute_envelope
    ckpt_args["device"] = args.device

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
        "limit_prompts": args.limit_prompts,
        "gen_config": gen_config,
        "gen_config_sha256": cfg_sha,
        "ckpt_args": ckpt_args,
        "rank": args.rank,
        "world_size": args.world_size,
        "cache_pipe": args.cache_pipe,
        "mode_batched": args.mode_batched or args.cache_pipe,
        "manifest_writer_schema_version": _MANIFEST_WRITER_SCHEMA_VERSION,
    }
    _mw_atomic_write_json(run_dir / "run_manifest.pre.json", run_manifest_pre)

    # 7. orchestration loop
    if args.cache_pipe and not args.mode_batched:
        print(
            "[heldout_regen] --cache-pipe implies --mode-batched (no DiffSynth LoRA "
            "detach path); enabling mode-batched iteration.",
            flush=True,
        )
        mode_batched = True
    else:
        mode_batched = args.mode_batched

    results = regen_all(
        selections=selections,
        gen_config_bytes=gen_config_bytes,
        out_root=run_dir,
        adapter=adapter,
        ckpt_args=ckpt_args,
        resume=not args.no_resume,
        rank=args.rank,
        world_size=args.world_size,
        mode_batched=mode_batched,
    )

    run_manifest_post = dict(run_manifest_pre)
    run_manifest_post["results_count"] = len(results)
    run_manifest_post["complete"] = all(r.get("complete") for r in results)
    run_manifest_post["results"] = results
    _mw_atomic_write_json(run_dir / "run_manifest.json", run_manifest_post)
    print(
        f"[done] regenerated {len(results)} prompts (rank {args.rank}/{args.world_size}); "
        f"manifest at {run_dir / 'run_manifest.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
