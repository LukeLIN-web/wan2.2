"""Wan2.2-I2V-A14B Direct I2V eval-harness smoke (M5 / AC-7.0).

Runs ONE prompt twice under one byte-identical generation_config:

  1. baseline: original sharded high-noise expert + frozen original
     low-noise expert (no LoRA on either side).
  2. trained:  original sharded high-noise expert + LoRA adapter on
     pipe.dit (high-noise only) + frozen original low-noise expert.

Goal is generation-pipeline validation: each run must complete without
OOM, dtype, or device errors. Each run writes the sampled video and a
run manifest to a timestamped output subdir. After both runs complete,
the script asserts that the JSON-serialized generation_config is
byte-equal across the two manifests (AC-7.2 byte-identical contract).

Hard contracts (videodpo/humanize/i2v.md):

  * recipe_id pinned at 6bef6e104cdd3442 (AC-3.1, line 26).
  * switch_DiT_boundary = 0.9, fps = 16, frame_num = 81 (AC-3.5).
  * LoRA is applied to pipe.dit (high-noise) only; pipe.dit2 (low-noise)
    is frozen and untouched.
  * Conditioning image must come from the canonical T2 mapping at
    <T0_T3_ROOT>/t2/image_manifest.json for the 42-prompt heldout pass
    (AC-7.1). The smoke entrypoint takes --cond-image directly so the
    caller is responsible for honouring that contract.

Repository boundary: this file lives in videodpoWan/humanize/dpo_v0/
per CLAUDE.md / i2v.md "Repository Boundary".
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
from typing import Any, Literal, Optional

import torch
from PIL import Image


HERE = pathlib.Path(__file__).resolve().parent


def _resolve_recipes_dir() -> pathlib.Path:
    """Walk parents until ``recipes/recipe_id`` is found; tolerates flat or
    eval/-subdir layouts."""
    for cand in (HERE / "recipes", HERE.parent / "recipes"):
        if (cand / "recipe_id").exists():
            return cand
    return HERE.parent / "recipes"


RECIPES_DIR = _resolve_recipes_dir()
DPO_ROOT = RECIPES_DIR.parent  # humanize/dpo_v0/
# AC-3.1: frozen recipe_id; any drift halts before any forward pass.
EXPECTED_RECIPE_ID = "6bef6e104cdd3442"

# AC-3.5: hard scheduler/codec pins shared with the trainer.
SWITCH_DIT_BOUNDARY = 0.9
NUM_FRAMES = 81
FPS = 16

# DEC-6: canonical envelope enum shared across trainer / smoke / M6.
COMPUTE_ENVELOPES_CANONICAL = (
    "single_gpu",
    "dpo_multi_gpu_ddp",
    "dpo_multi_gpu_zero2",
    "multi_gpu_inference_seed_parallel",
)

Mode = Literal["both", "baseline", "trained"]

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# Pin / provenance helpers
# ---------------------------------------------------------------------------


def assert_recipe_pin(recipes_dir: pathlib.Path = RECIPES_DIR,
                      expected: str = EXPECTED_RECIPE_ID) -> str:
    """Recompute sha256(canonical_yaml.bytes)[:16] and 3-way assert against
    on-disk pin and ``expected``."""
    yaml_bytes = (recipes_dir / "wan22_i2v_a14b__round2_v0.yaml").read_bytes()
    fresh = hashlib.sha256(yaml_bytes).hexdigest()[:16]
    on_disk = (recipes_dir / "recipe_id").read_text(encoding="ascii").strip()
    if not (fresh == on_disk == expected):
        raise RuntimeError(
            f"recipe_id pin drift: fresh={fresh}, on_disk={on_disk}, "
            f"expected={expected}"
        )
    return on_disk


def file_sha256(path: pathlib.Path, buf: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


# Sidecar-cached aggregated shard sha256, shared via humanize/dpo_v0/file_sha_cache.py.
# Layout-tolerant import: file_sha_cache.py sits next to humanize/dpo_v0/ regardless
# of whether this file is at humanize/dpo_v0/ or humanize/dpo_v0/eval/.
def _import_file_sha_cache():
    import importlib.util as _ilu
    here = pathlib.Path(__file__).resolve()
    for cand in (here.parent, here.parent.parent):
        f = cand / "file_sha_cache.py"
        if f.exists():
            spec = _ilu.spec_from_file_location("_videodpo_file_sha_cache", f)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("file_sha_cache.py not found beside inference_smoke.py")


sharded_ckpt_sha = _import_file_sha_cache().cached_sharded_ckpt_sha  # noqa: E305


def code_commit_id() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE), stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def machine_internal_ip_tail() -> str:
    try:
        ip = socket.gethostbyname(socket.gethostname())
        return ip.rsplit(".", 1)[-1]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Upstream shard layout
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UpstreamPaths:
    """Resolved canonical paths for the I2V-A14B upstream shards.

    The upstream root is /shared/user63/workspace/data/Wan/Wan2.2-I2V-A14B/
    on the shared filesystem; that path is the same one the loader's
    high_noise / low_noise manifests pin against (per-shard SHA recorded
    under the canonical hash rule, AC-3.4).
    """

    upstream_root: pathlib.Path

    @property
    def high_noise_shards(self) -> list[pathlib.Path]:
        return sorted(
            (self.upstream_root / "high_noise_model").glob(
                "diffusion_pytorch_model-*-of-*.safetensors"
            )
        )

    @property
    def low_noise_shards(self) -> list[pathlib.Path]:
        return sorted(
            (self.upstream_root / "low_noise_model").glob(
                "diffusion_pytorch_model-*-of-*.safetensors"
            )
        )

    @property
    def t5(self) -> pathlib.Path:
        return self.upstream_root / "models_t5_umt5-xxl-enc-bf16.pth"

    @property
    def vae(self) -> pathlib.Path:
        return self.upstream_root / "Wan2.1_VAE.pth"

    @property
    def tokenizer_dir(self) -> pathlib.Path:
        return self.upstream_root / "google" / "umt5-xxl"

    def assert_present(self) -> None:
        missing = []
        for label, p in [
            ("high_noise_model/", self.upstream_root / "high_noise_model"),
            ("low_noise_model/", self.upstream_root / "low_noise_model"),
            ("t5", self.t5),
            ("vae", self.vae),
            ("tokenizer_dir", self.tokenizer_dir),
        ]:
            if not p.exists():
                missing.append(f"{label} -> {p}")
        if missing:
            raise FileNotFoundError(
                "Missing upstream files:\n  " + "\n  ".join(missing)
            )
        if len(self.high_noise_shards) != 6 or len(self.low_noise_shards) != 6:
            raise RuntimeError(
                "Expected 6 shards per expert; got "
                f"high={len(self.high_noise_shards)} "
                f"low={len(self.low_noise_shards)}"
            )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline(upstream: UpstreamPaths,
                   torch_dtype: torch.dtype,
                   device: str):
    """Construct a Wan2.2-I2V-A14B WanVideoPipeline from canonical sharded
    weights. Returns the pipeline; pipe.dit is high-noise, pipe.dit2 is
    low-noise (model_pool.fetch_model preserves load order).
    """
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    high_paths = [str(p) for p in upstream.high_noise_shards]
    low_paths = [str(p) for p in upstream.low_noise_shards]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(path=high_paths),
            ModelConfig(path=low_paths),
            ModelConfig(path=str(upstream.t5)),
            ModelConfig(path=str(upstream.vae)),
        ],
        tokenizer_config=ModelConfig(path=str(upstream.tokenizer_dir)),
    )
    return pipe


def attach_lora(pipe, lora_path: pathlib.Path) -> None:
    """Attach the trained LoRA adapter to pipe.dit (high-noise) only.

    The low-noise expert (pipe.dit2) is never touched. The trainer freezes
    low-noise and asserts byte-equality post-run (AC-6); the smoke run
    mirrors that contract by leaving pipe.dit2 alone.
    """
    if pipe.dit is None:
        raise RuntimeError("pipe.dit is None; cannot attach LoRA")
    pipe.load_lora(pipe.dit, str(lora_path), alpha=1.0)


# ---------------------------------------------------------------------------
# Generation config (shared across both runs)
# ---------------------------------------------------------------------------


def build_generation_config(args: argparse.Namespace) -> dict:
    """Construct the canonical generation_config dict.

    The SAME dict is used to drive both runs, and the SAME JSON
    serialization is written into both run manifests; the script then
    asserts byte-equality between the two stamped configs (AC-7.2 /
    rl2 defensive note: do not hand-copy a config dict).
    """
    return {
        "seed": args.seed,
        "sampler": "wan_default_flow_match",
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "dtype": str(args.torch_dtype),
        "switch_DiT_boundary": args.switch_DiT_boundary,
        "tiled": True,
        "tile_size": [30, 52],
        "tile_stride": [15, 26],
        "rand_device": "cpu",
        "judge_preprocessing": "phygroundata.md_section8",
    }


def serialize_generation_config(cfg: dict) -> bytes:
    """Canonical JSON encoding for byte-identical comparison.

    sort_keys=True + ensure_ascii=False + UTF-8 + LF-only. This is the
    only serialization we trust for the byte-equality assertion.
    """
    return (json.dumps(cfg, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_one_sample(
    *,
    mode: Mode,
    out_dir: pathlib.Path,
    upstream: UpstreamPaths,
    torch_dtype: torch.dtype,
    device: str,
    prompt: str,
    cond_image_path: pathlib.Path,
    generation_config: dict,
    generation_config_bytes: bytes,
    lora_adapter_path: Optional[pathlib.Path],
    compute_envelope: str,
    recipe_id: str,
    high_noise_sha: Optional[str] = None,
    low_noise_sha: Optional[str] = None,
    pipe: Any = None,
    lora_already_attached: bool = False,
    lora_sha: Optional[str] = None,
) -> dict[str, Any]:
    """Run one inference sample and write video + manifest.

    Public Python API used by the CLI entry point and by M6's heldout regen
    orchestrator. Returns the manifest dict so the caller can read back
    ``ckpt_shas`` (AC-7.3 load-side ground truth) without re-reading the file.

    ``mode`` is ``"baseline"`` (no LoRA) or ``"trained"`` (LoRA on pipe.dit).
    On-disk layout is ``out_dir/<mode>/<ts>/...``.

    ``high_noise_sha`` / ``low_noise_sha`` / ``lora_sha`` are optional caches;
    when omitted they are computed inline.

    ``pipe`` lets a multi-call caller (e.g. heldout regen with
    ``world_size << len(prompts)``) build the ~95 GB pipeline once and reuse
    it. When provided, the caller MUST build it from the same ``upstream`` /
    ``torch_dtype`` / ``device`` recorded here, and track LoRA attach state
    via ``lora_already_attached`` (DiffSynth has no clean detach path; rebuild
    is required between different LoRA paths or trained→baseline).
    """
    if mode not in ("baseline", "trained"):
        raise ValueError(f"mode must be 'baseline' or 'trained'; got {mode!r}")
    if mode == "trained" and lora_adapter_path is None:
        raise ValueError("mode='trained' requires lora_adapter_path")
    if mode == "baseline" and lora_adapter_path is not None:
        raise ValueError(
            "mode='baseline' must not be passed a lora_adapter_path "
            f"(got {lora_adapter_path}); pass mode='trained' instead"
        )
    if lora_already_attached and lora_adapter_path is None:
        raise ValueError(
            "lora_already_attached=True requires lora_adapter_path; got None"
        )
    if pipe is None and lora_already_attached:
        raise ValueError(
            "lora_already_attached=True is meaningless without a cached pipe"
        )

    run_dir = out_dir / mode / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    if high_noise_sha is None:
        high_noise_sha = sharded_ckpt_sha(upstream.high_noise_shards)
    if low_noise_sha is None:
        low_noise_sha = sharded_ckpt_sha(upstream.low_noise_shards)

    if pipe is None:
        print(f"[{mode}] building pipeline ...", flush=True)
        pipe = build_pipeline(upstream, torch_dtype=torch_dtype, device=device)
    else:
        print(f"[{mode}] reusing cached pipeline (skip build)", flush=True)

    if lora_adapter_path is not None:
        if lora_sha is None:
            lora_sha = file_sha256(lora_adapter_path)
        if lora_already_attached:
            print(
                f"[{mode}] LoRA {lora_adapter_path} sha={lora_sha[:12]}... "
                f"already attached by caller; skip attach",
                flush=True,
            )
        else:
            print(f"[{mode}] attaching LoRA {lora_adapter_path} sha={lora_sha[:12]}...",
                  flush=True)
            attach_lora(pipe, lora_adapter_path)

    print(f"[{mode}] loading conditioning image {cond_image_path}", flush=True)
    cond_image = Image.open(str(cond_image_path)).convert("RGB")

    print(f"[{mode}] generating ...", flush=True)
    video = pipe(
        prompt=prompt,
        negative_prompt=generation_config["negative_prompt"],
        input_image=cond_image,
        seed=generation_config["seed"],
        height=generation_config["height"],
        width=generation_config["width"],
        num_frames=generation_config["num_frames"],
        cfg_scale=generation_config["cfg_scale"],
        switch_DiT_boundary=generation_config["switch_DiT_boundary"],
        num_inference_steps=generation_config["num_inference_steps"],
        tiled=generation_config["tiled"],
        tile_size=tuple(generation_config["tile_size"]),
        tile_stride=tuple(generation_config["tile_stride"]),
        rand_device=generation_config["rand_device"],
    )

    video_path = run_dir / "video.mp4"
    from diffsynth.utils.data import save_video
    save_video(video, str(video_path), fps=generation_config["fps"], quality=5)

    # AC-7.3: baseline stamps high_noise + low_noise SHAs; trained additionally
    # stamps lora_adapter_sha. All three come from the load-side, so M6's
    # `_extract_ckpt_shas` can trust the manifest as ground truth even when the
    # on-disk path was rewritten by a deploy step.
    cond_image_sha = file_sha256(cond_image_path)
    ckpt_shas: dict[str, Optional[str]] = {
        "high_noise_base_sha256": high_noise_sha,
        "low_noise_frozen_sha256": low_noise_sha,
        "lora_adapter_sha256": lora_sha if mode == "trained" else None,
    }
    manifest = {
        "schema_version": 1,
        "mode": mode,
        # `run_label` retained for backward compatibility with older consumers.
        "run_label": mode,
        "run_dir": str(run_dir),
        "timestamp_utc": run_dir.name,
        "code_commit_id": code_commit_id(),
        "machine_internal_ip_tail": machine_internal_ip_tail(),
        "compute_envelope": compute_envelope,
        "recipe_id": recipe_id,
        "switch_DiT_boundary_raw": int(SWITCH_DIT_BOUNDARY * 1000),
        "upstream_root": str(upstream.upstream_root),
        "high_noise_shards": [p.name for p in upstream.high_noise_shards],
        "low_noise_shards": [p.name for p in upstream.low_noise_shards],
        "lora_adapter_path": str(lora_adapter_path) if lora_adapter_path else None,
        "lora_adapter_sha256": lora_sha,
        "ckpt_shas": ckpt_shas,
        "ckpt_paths": {
            "high_noise_base": str(upstream.upstream_root / "high_noise_model"),
            "low_noise_frozen": str(upstream.upstream_root / "low_noise_model"),
            "lora_adapter": str(lora_adapter_path) if lora_adapter_path else None,
        },
        "gen_config_sha256": hashlib.sha256(generation_config_bytes).hexdigest(),
        "prompt": prompt,
        "cond_image_path": str(cond_image_path),
        "cond_image_sha256": cond_image_sha,
        "out_video_path": str(video_path),
        "torch_version": torch.__version__,
        "torch_dtype": str(torch_dtype),
        "device": device,
        "generation_config": generation_config,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    # The byte-identical generation_config is written separately so the
    # post-run cross-comparison reads the exact bytes the script
    # emitted, not a re-serialization of the manifest dict.
    (run_dir / "generation_config.json").write_bytes(generation_config_bytes)
    print(f"[{mode}] done -> {run_dir}", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Argparse + entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M5 / AC-7.0 eval-harness smoke for Wan2.2-I2V-A14B Direct I2V "
            "DPO v0. Runs one prompt twice (baseline + trained) under one "
            "shared generation_config and asserts byte-equality."
        )
    )
    parser.add_argument(
        "--upstream",
        type=pathlib.Path,
        default=pathlib.Path("/shared/user63/workspace/data/Wan/Wan2.2-I2V-A14B"),
        help="Canonical upstream root (default: shared cluster path).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "baseline", "trained"],
        default="both",
        help=(
            "'both' runs baseline followed by trained under one shared "
            "generation_config (smoke). 'baseline' / 'trained' run a single "
            "sample for callers driving them as separate subprocess invocations."
        ),
    )
    parser.add_argument(
        "--lora-adapter",
        type=pathlib.Path,
        default=None,
        help="LoRA safetensors. Required for --mode in {both, trained}; rejected with baseline.",
    )
    parser.add_argument(
        "--low-noise-ckpt",
        type=pathlib.Path,
        default=None,
        help="Optional explicit frozen low-noise expert path; defaults to <upstream>/low_noise_model/.",
    )
    parser.add_argument(
        "--gen-config-json",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to a pre-built generation_config.json. When given, the bytes "
            "are loaded verbatim as the byte-identical config; argparse generation "
            "knobs are ignored to avoid drift."
        ),
    )
    parser.add_argument(
        "--recipe-yaml",
        type=pathlib.Path,
        default=None,
        help="Recipe YAML; defaults to <this_dir>/recipes/wan22_i2v_a14b__round2_v0.yaml.",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--cond-image",
        type=pathlib.Path,
        required=True,
        help=(
            "Conditioning image. For the 42-prompt heldout pass this MUST "
            "come from <T0_T3_ROOT>/t2/image_manifest.json (AC-7.1)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Output dir; baseline/<ts>/ and trained/<ts>/ are created underneath.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument(
        "--switch-DiT-boundary",
        type=float,
        default=SWITCH_DIT_BOUNDARY,
        help="Frozen at 0.9 by AC-3.5; flag exists only for explicit override audit.",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda"
    )
    parser.add_argument(
        "--compute-envelope",
        type=str,
        default="single_gpu",
        choices=list(COMPUTE_ENVELOPES_CANONICAL),
        help="DEC-6 envelope tag stamped per-run. Smoke is single_gpu by contract.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate args + pin + upstream + config, then exit before pipeline build.",
    )
    args = parser.parse_args(argv)
    args.torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    if args.switch_DiT_boundary != SWITCH_DIT_BOUNDARY:
        raise SystemExit(
            f"--switch-DiT-boundary must be {SWITCH_DIT_BOUNDARY} (AC-3.5); "
            f"got {args.switch_DiT_boundary}"
        )
    # AC-3.5 pins num_frames=81 / fps=16 for the canonical eval, but the eval
    # pipeline supplies these via --gen-config-json (which bypasses argparse).
    # Argparse defaults match the pin; we don't assert equality so dev/smoke
    # runs can lower num_frames or fps. Actual values are stamped per-run.
    if args.num_frames < 1:
        raise SystemExit(f"--num-frames must be >= 1; got {args.num_frames}")
    if (args.num_frames - 1) % 4 != 0:
        raise SystemExit(
            f"--num-frames must satisfy (n - 1) %% 4 == 0 (Wan temporal "
            f"latent factor); got {args.num_frames}. Valid values include "
            f"1, 5, 9, ..., 41, 81."
        )
    if args.fps < 1:
        raise SystemExit(f"--fps must be >= 1; got {args.fps}")

    # Mode-conditioned LoRA validation: baseline rejects --lora-adapter so a
    # stray flag never silently taints a baseline run.
    if args.mode in ("both", "trained") and args.lora_adapter is None:
        raise SystemExit(
            f"--mode={args.mode} requires --lora-adapter <path>; got None"
        )
    if args.mode == "baseline" and args.lora_adapter is not None:
        raise SystemExit(
            "--mode=baseline must not be passed --lora-adapter "
            f"(got {args.lora_adapter}); use --mode=trained for LoRA."
        )
    return args


def _resolve_recipe_pin(args: argparse.Namespace) -> str:
    """Resolve recipe_id from --recipe-yaml (if given) or default location."""
    if args.recipe_yaml is not None:
        # Caller supplied an explicit YAML path; assert against its dir's
        # `recipe_id` sidecar so the 3-way drift check still applies.
        recipes_dir = args.recipe_yaml.parent
    else:
        recipes_dir = RECIPES_DIR
    return assert_recipe_pin(recipes_dir=recipes_dir)


def _load_generation_config(args: argparse.Namespace) -> tuple[dict, bytes]:
    """Pick generation_config either from --gen-config-json or argparse knobs.

    When --gen-config-json is given, the file bytes are loaded verbatim and
    are the single source of truth for the byte-identical contract; the
    JSON dict is parsed for the actual runtime params. When omitted, the
    config is built from argparse knobs and serialized via
    `serialize_generation_config(...)` (back-compat with luke's round-1
    entry point and the smoke / `--mode=both` use case).
    """
    if args.gen_config_json is not None:
        gen_config_bytes = args.gen_config_json.read_bytes()
        gen_config = json.loads(gen_config_bytes)
        return gen_config, gen_config_bytes
    gen_config = build_generation_config(args)
    return gen_config, serialize_generation_config(gen_config)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    recipe_id = _resolve_recipe_pin(args)
    print(f"[smoke] recipe_id pin OK: {recipe_id}", flush=True)

    upstream = UpstreamPaths(upstream_root=args.upstream)
    upstream.assert_present()
    print(f"[smoke] upstream present: {args.upstream}", flush=True)

    if args.lora_adapter is not None and not args.lora_adapter.exists():
        raise SystemExit(f"--lora-adapter does not exist: {args.lora_adapter}")
    if not args.cond_image.exists():
        raise SystemExit(f"--cond-image does not exist: {args.cond_image}")

    generation_config, generation_config_bytes = _load_generation_config(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "generation_config.json").write_bytes(generation_config_bytes)

    if args.dry_run:
        print(
            "[smoke] --dry-run: pin / paths / config OK; exiting before pipeline",
            flush=True,
        )
        return 0

    # Cache the sharded base hashes once across both modes. Computing each
    # ckpt SHA traverses ~28 GB of safetensors; doing it twice would burn
    # ~6-10 minutes of disk IO for no value (the bytes are identical across
    # baseline and trained -- LoRA applies in-RAM only).
    high_sha = sharded_ckpt_sha(upstream.high_noise_shards)
    low_sha = sharded_ckpt_sha(upstream.low_noise_shards)

    baseline_manifest: Optional[dict[str, Any]] = None
    trained_manifest: Optional[dict[str, Any]] = None

    if args.mode in ("both", "baseline"):
        baseline_manifest = run_one_sample(
            mode="baseline",
            out_dir=args.out_dir,
            upstream=upstream,
            torch_dtype=args.torch_dtype,
            device=args.device,
            prompt=args.prompt,
            cond_image_path=args.cond_image,
            generation_config=generation_config,
            generation_config_bytes=generation_config_bytes,
            lora_adapter_path=None,
            compute_envelope=args.compute_envelope,
            recipe_id=recipe_id,
            high_noise_sha=high_sha,
            low_noise_sha=low_sha,
        )
    if args.mode in ("both", "trained"):
        trained_manifest = run_one_sample(
            mode="trained",
            out_dir=args.out_dir,
            upstream=upstream,
            torch_dtype=args.torch_dtype,
            device=args.device,
            prompt=args.prompt,
            cond_image_path=args.cond_image,
            generation_config=generation_config,
            generation_config_bytes=generation_config_bytes,
            lora_adapter_path=args.lora_adapter,
            compute_envelope=args.compute_envelope,
            recipe_id=recipe_id,
            high_noise_sha=high_sha,
            low_noise_sha=low_sha,
        )

    if args.mode == "both":
        # AC-7.2: dual-run smoke must verify byte-identical config across the
        # two stamped run dirs (not the two manifest dicts; the comparison
        # MUST be byte-level since the test failure mode is hand-copying or
        # accidental dict re-serialization in different orders).
        assert baseline_manifest is not None and trained_manifest is not None
        baseline_dir = pathlib.Path(baseline_manifest["run_dir"])
        trained_dir = pathlib.Path(trained_manifest["run_dir"])
        baseline_cfg = (baseline_dir / "generation_config.json").read_bytes()
        trained_cfg = (trained_dir / "generation_config.json").read_bytes()
        if baseline_cfg != trained_cfg:
            raise RuntimeError(
                "generation_config bytes diverge between baseline and "
                f"trained runs (baseline={len(baseline_cfg)}B, "
                f"trained={len(trained_cfg)}B). AC-7.2 byte-identical "
                "contract violated."
            )
        print(
            f"[smoke] generation_config byte-identical across runs "
            f"({len(baseline_cfg)}B). AC-7.0 smoke complete.",
            flush=True,
        )
    return 0


__all__ = [
    "EXPECTED_RECIPE_ID",
    "SWITCH_DIT_BOUNDARY",
    "NUM_FRAMES",
    "FPS",
    "COMPUTE_ENVELOPES_CANONICAL",
    "Mode",
    "UpstreamPaths",
    "assert_recipe_pin",
    "file_sha256",
    "sharded_ckpt_sha",
    "build_generation_config",
    "serialize_generation_config",
    "run_one_sample",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
