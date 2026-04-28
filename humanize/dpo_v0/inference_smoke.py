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
from typing import Optional

import torch
from PIL import Image


HERE = pathlib.Path(__file__).resolve().parent
RECIPES_DIR = HERE / "recipes"
# AC-3.1 (videodpo/humanize/i2v.md line 26): recipe_id is frozen at this
# value across the parent plan and the i2v plan; any drift halts before
# any forward pass.
EXPECTED_RECIPE_ID = "6bef6e104cdd3442"

# AC-3.5: hard scheduler/codec pins shared with the trainer.
SWITCH_DIT_BOUNDARY = 0.9
NUM_FRAMES = 81
FPS = 16

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
    """Recompute sha256(canonical_yaml.bytes)[:16] and assert pin equality.

    Mirrors the trainer's pin assertion so the smoke run shares the same
    provenance contract.
    """
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
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def run_one_sample(
    *,
    run_label: str,
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
) -> pathlib.Path:
    """Run one inference sample and write video + manifest.

    Returns the run dir (timestamped).
    """
    run_dir = out_dir / run_label / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_label}] building pipeline ...", flush=True)
    pipe = build_pipeline(upstream, torch_dtype=torch_dtype, device=device)

    lora_sha = None
    if lora_adapter_path is not None:
        lora_sha = file_sha256(lora_adapter_path)
        print(f"[{run_label}] attaching LoRA {lora_adapter_path} sha={lora_sha[:12]}...",
              flush=True)
        attach_lora(pipe, lora_adapter_path)

    print(f"[{run_label}] loading conditioning image {cond_image_path}", flush=True)
    cond_image = Image.open(str(cond_image_path)).convert("RGB")

    print(f"[{run_label}] generating ...", flush=True)
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

    # Per-shard SHAs are recorded by the loader's manifest at
    # humanize/dpo_v0/loader/out/<ts>/{high_noise,low_noise}/manifest.json;
    # we record the upstream root here and let the cross-reference happen
    # via the loader manifest path stored alongside.
    cond_image_sha = file_sha256(cond_image_path)
    manifest = {
        "schema_version": 1,
        "run_label": run_label,
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
        "prompt": prompt,
        "cond_image_path": str(cond_image_path),
        "cond_image_sha256": cond_image_sha,
        "video_path": str(video_path),
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
    print(f"[{run_label}] done -> {run_dir}", flush=True)
    return run_dir


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
        "--lora-adapter",
        type=pathlib.Path,
        required=True,
        help=(
            "Path to the trained LoRA safetensors. Produced by "
            "train_dpo_i2v.py at the end of M3/M4. Used for the trained "
            "run only; the baseline run never sees it."
        ),
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--cond-image",
        type=pathlib.Path,
        required=True,
        help=(
            "Conditioning image. For the 42-prompt heldout pass this MUST "
            "come from <T0_T3_ROOT>/t2/image_manifest.json (AC-7.1); the "
            "smoke entrypoint trusts the caller."
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
        choices=["single_gpu", "dpo_multi_gpu_zero2"],
        help="Stamped per-run; DEC-6. Smoke is single_gpu by contract.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate args + recipe pin + upstream presence + serialize the "
            "generation_config, then exit before constructing the pipeline. "
            "Useful for unit-style checks without GPU."
        ),
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
    if args.num_frames != NUM_FRAMES:
        raise SystemExit(
            f"--num-frames must be {NUM_FRAMES} (AC-3.5); got {args.num_frames}"
        )
    if args.fps != FPS:
        raise SystemExit(f"--fps must be {FPS} (AC-3.5); got {args.fps}")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    recipe_id = assert_recipe_pin()
    print(f"[smoke] recipe_id pin OK: {recipe_id}", flush=True)

    upstream = UpstreamPaths(upstream_root=args.upstream)
    upstream.assert_present()
    print(f"[smoke] upstream present: {args.upstream}", flush=True)

    if not args.lora_adapter.exists():
        raise SystemExit(f"--lora-adapter does not exist: {args.lora_adapter}")
    if not args.cond_image.exists():
        raise SystemExit(f"--cond-image does not exist: {args.cond_image}")

    generation_config = build_generation_config(args)
    generation_config_bytes = serialize_generation_config(generation_config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "generation_config.json").write_bytes(generation_config_bytes)

    if args.dry_run:
        print("[smoke] --dry-run: pin / paths / config OK; exiting before pipeline", flush=True)
        return 0

    baseline_dir = run_one_sample(
        run_label="baseline",
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
    )
    trained_dir = run_one_sample(
        run_label="trained",
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
    )

    baseline_cfg = (baseline_dir / "generation_config.json").read_bytes()
    trained_cfg = (trained_dir / "generation_config.json").read_bytes()
    if baseline_cfg != trained_cfg:
        raise RuntimeError(
            "generation_config bytes diverge between baseline and trained runs "
            f"(baseline={len(baseline_cfg)}B, trained={len(trained_cfg)}B). "
            "AC-7.2 byte-identical contract violated."
        )
    print(
        f"[smoke] generation_config byte-identical across runs "
        f"({len(baseline_cfg)}B). AC-7.0 smoke complete.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
