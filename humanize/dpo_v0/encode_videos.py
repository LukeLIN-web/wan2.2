"""Encode tier_a (and optionally tier_b slice) videos into Wan 2.1 VAE latents.

Reads ``T3_subset.json`` for the pair list, reads ``post_t2_pair.json`` for
the winner / loser metadata, then for every (pair, role) loads the source
mp4 from ``/shared/user60/worldmodel/wmbench/data/videos/<dataset>/<filename>``,
applies the canonical preprocessing recipe (recipe_id ``6bef6e104cdd3442``),
encodes via the Wan 2.1 VAE bundled with the I2V-A14B upstream, and writes
the latent + a sidecar manifest entry under
``humanize/dpo_v0/latents/<UTC>/<expert>/<pair_id>__<role>.safetensors`` with
sibling ``manifest.jsonl`` carrying the recipe_id pin, vae sha256, video
sha256, normalized resolution, and per-frame stats.

Heldout-leak guard (rl2 acceptance ``8e40d34c`` #1): startup loads the
``T3_subset.json:heldout_excluded.scene_filenames`` blocklist; any video
whose ``filename`` matches that set raises before any encode call.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import pathlib
import sys
import time

import cv2
import numpy as np
import safetensors.torch
import torch

# Keep the loader's recipe pin in lockstep with this script.
HUMANIZE_DIR = pathlib.Path(__file__).resolve().parent
RECIPES_DIR = HUMANIZE_DIR / "recipes"
EXPECTED_RECIPE_ID = "6bef6e104cdd3442"
T3_SUBSET_JSON = pathlib.Path(
    "/home/user1/.config/superpowers/worktrees/videodpoWan/rlcr-task-5/humanize/dpo_v0/T3_subset.json"
)
POST_T2_PAIR_JSON = pathlib.Path(
    "/home/user1/.config/superpowers/worktrees/videodpoWan/rlcr-task-5/humanize/dpo_v0/out/20260427T201113Z/t2/post_t2_pair.json"
)
VIDEO_ROOT = pathlib.Path("/shared/user60/worldmodel/wmbench/data/videos")
UPSTREAM = pathlib.Path("/shared/user63/workspace/data/Wan/Wan2.2-I2V-A14B")
VAE_PATH = UPSTREAM / "Wan2.1_VAE.pth"

# Recipe values (pinned via recipe_id; do not edit without bumping recipe_id).
TARGET_LANDSCAPE = (832, 480)  # (W, H)
TARGET_PORTRAIT = (480, 832)
FRAME_NUM = 81
PAD_COLOR_BT709 = (0, 0, 0)


def _file_sha256(path: pathlib.Path, buf: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_recipe_pin() -> str:
    yaml_bytes = (RECIPES_DIR / "wan22_i2v_a14b__round2_v0.yaml").read_bytes()
    fresh = hashlib.sha256(yaml_bytes).hexdigest()[:16]
    on_disk = (RECIPES_DIR / "recipe_id").read_text(encoding="ascii").strip()
    if not (fresh == on_disk == EXPECTED_RECIPE_ID):
        raise ValueError(
            f"recipe pin drift: fresh={fresh}, on_disk={on_disk}, expected={EXPECTED_RECIPE_ID}"
        )
    return on_disk


def _load_heldout_blocklist() -> set[str]:
    """Load the heldout filename set from T3_subset.json (AC-4 guard)."""
    subset = json.loads(T3_SUBSET_JSON.read_bytes())
    return set(subset["heldout_excluded"]["scene_filenames"])


def _resolve_target(width: int, height: int) -> tuple[int, int]:
    if width >= height:
        return TARGET_LANDSCAPE
    return TARGET_PORTRAIT


def _letterbox_pad(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize ``frame`` keeping aspect ratio, pad to (target_h, target_w) with black."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=PAD_COLOR_BT709,
    )
    return padded


def _read_first_n_frames(video_path: pathlib.Path, frame_num: int, target_w: int, target_h: int) -> np.ndarray:
    """Return [T, H, W, 3] uint8 RGB array of the first ``frame_num`` frames after letterbox pad."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < frame_num:
            ok, frame = cap.read()
            if not ok:
                break
            # cv2 reads BGR; recipe says color_space = bt709-tv-range, codec_normalization = yuv420p_to_rgb_bt709_full.
            # cv2's default decode is full-range BGR; we convert to RGB and trust ffmpeg's BT.709 mapping.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            padded = _letterbox_pad(rgb, target_w, target_h)
            frames.append(padded)
    finally:
        cap.release()
    if len(frames) < frame_num:
        raise RuntimeError(
            f"video has only {len(frames)} frames, recipe requires {frame_num}: {video_path}"
        )
    arr = np.stack(frames, axis=0)
    return arr  # [T, H, W, 3] uint8 RGB


def _frames_to_tensor(frames_uint8: np.ndarray) -> torch.Tensor:
    """[T, H, W, 3] uint8 RGB -> [3, T, H, W] float32 in [-1, 1]."""
    f = torch.from_numpy(frames_uint8).float() / 127.5 - 1.0
    return f.permute(3, 0, 1, 2).contiguous()


def _resolve_video_path(record_role: dict) -> pathlib.Path:
    return VIDEO_ROOT / record_role["dataset"] / record_role["filename"]


@dataclasses.dataclass
class EncodedEntry:
    pair_id: str
    role: str  # "winner" or "loser"
    dataset: str
    filename: str
    source_video_path: str
    source_video_sha256: str
    target_resolution: list[int]
    frame_num: int
    latent_shape: list[int]
    latent_dtype: str
    latent_path: str
    recipe_id: str
    vae_sha256: str
    encode_seconds: float


def encode_pair_role(
    record: dict, role: str, vae, out_dir: pathlib.Path, recipe_id: str, vae_sha: str,
    heldout_blocklist: set[str], device: torch.device, dtype: torch.dtype,
) -> EncodedEntry:
    info = record[role]
    filename = info["filename"]
    if filename in heldout_blocklist:
        raise RuntimeError(
            f"AC-4 violation: heldout scene encode attempted: {filename} (pair {record['pair_id']}, role {role})"
        )
    src = _resolve_video_path(info)
    if not src.exists():
        raise FileNotFoundError(f"missing source video: {src}")
    cap = cv2.VideoCapture(str(src))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    target_w, target_h = _resolve_target(width, height)

    src_sha = _file_sha256(src)
    frames = _read_first_n_frames(src, FRAME_NUM, target_w, target_h)
    video_tensor = _frames_to_tensor(frames).to(device=device, dtype=dtype)
    t0 = time.time()
    latents = vae.encode([video_tensor])  # [3, T, H, W] -> [16, T', H', W']
    elapsed = time.time() - t0
    latent = latents[0].detach().cpu().contiguous()  # float (returned as float32 in encode)

    safe_pair = record["pair_id"].replace("/", "_")
    out_path = out_dir / f"{safe_pair}__{role}.safetensors"
    safetensors.torch.save_file({"latent": latent.to(torch.bfloat16)}, str(out_path))
    return EncodedEntry(
        pair_id=record["pair_id"],
        role=role,
        dataset=info["dataset"],
        filename=filename,
        source_video_path=str(src),
        source_video_sha256=src_sha,
        target_resolution=[target_w, target_h],
        frame_num=FRAME_NUM,
        latent_shape=list(latent.shape),
        latent_dtype="bfloat16",
        latent_path=str(out_path),
        recipe_id=recipe_id,
        vae_sha256=vae_sha,
        encode_seconds=round(elapsed, 3),
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--tier",
        choices=["tier_a", "tier_b_first_n", "tier_b_round4_1k"],
        default="tier_a",
        help="tier_b_round4_1k reads pair_ids from --subset-pair-ids-json's "
             "'tier_b_round4_1k.pair_ids' field (round-4 task #19 output) and "
             "asserts pair_ids sha256[:16] == --pair-ids-sha256-pin.",
    )
    ap.add_argument("--tier-b-n", type=int, default=200, help="if --tier tier_b_first_n, encode the first N pair_ids")
    ap.add_argument(
        "--subset-pair-ids-json",
        type=pathlib.Path,
        default=None,
        help="Required when --tier tier_b_round4_1k: path to T3_round4_tier_b_1k.json (round-4 #19 output).",
    )
    ap.add_argument(
        "--pair-ids-sha256-pin",
        type=str,
        default=None,
        help="Required when --tier tier_b_round4_1k: expected sha256[:16] of newline-canonical pair_ids.",
    )
    ap.add_argument("--out-root", type=pathlib.Path, default=HUMANIZE_DIR / "latents")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args(argv[1:])

    recipe_id = _read_recipe_pin()
    print(f"recipe_id pin OK: {recipe_id}")
    blocklist = _load_heldout_blocklist()
    print(f"heldout blocklist size: {len(blocklist)}")

    subset = json.loads(T3_SUBSET_JSON.read_bytes())
    post_t2 = {r["pair_id"]: r for r in json.loads(POST_T2_PAIR_JSON.read_bytes())}
    if args.tier == "tier_a":
        pair_ids = list(subset["tier_a"]["pair_ids"])
    elif args.tier == "tier_b_first_n":
        pair_ids = list(subset["tier_b"]["pair_ids"])[: args.tier_b_n]
    else:  # tier_b_round4_1k
        if args.subset_pair_ids_json is None or args.pair_ids_sha256_pin is None:
            raise SystemExit(
                "--tier tier_b_round4_1k requires --subset-pair-ids-json AND --pair-ids-sha256-pin"
            )
        round4_data = json.loads(args.subset_pair_ids_json.read_bytes())
        pair_ids = list(round4_data["tier_b_round4_1k"]["pair_ids"])
        canonical = ("\n".join(pair_ids) + "\n").encode("utf-8")
        fresh = hashlib.sha256(canonical).hexdigest()[:16]
        if fresh != args.pair_ids_sha256_pin:
            raise SystemExit(
                f"pair_ids pin drift: fresh={fresh}, expected={args.pair_ids_sha256_pin}"
            )
        print(f"pair_ids pin OK: {fresh} (n_pairs={len(pair_ids)})")
    print(f"selected tier={args.tier}, pairs={len(pair_ids)}")

    print(f"hashing VAE at {VAE_PATH} ...")
    vae_sha = _file_sha256(VAE_PATH)
    print(f"vae_sha256 = {vae_sha}")

    dtype_t = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print(f"loading Wan2_1_VAE on {device} dtype={args.dtype} ...")
    sys.path.insert(0, "/shared/user60/worldmodel/rlvideo/videodpoWan")
    from wan.modules.vae2_1 import Wan2_1_VAE
    vae = Wan2_1_VAE(z_dim=16, vae_pth=str(VAE_PATH), dtype=dtype_t, device=str(device))
    print("vae ready")

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_root / ts / args.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    n_done = 0
    with manifest_path.open("wb") as f:
        for pair_id in pair_ids:
            record = post_t2[pair_id]
            for role in ("winner", "loser"):
                try:
                    entry = encode_pair_role(
                        record, role, vae, out_dir, recipe_id, vae_sha, blocklist, device, dtype_t,
                    )
                except Exception as e:
                    print(f"FAIL {pair_id}/{role}: {e}")
                    raise
                line = json.dumps(dataclasses.asdict(entry), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"
                f.write(line)
                f.flush()
                n_done += 1
                print(f"[{n_done}/{2 * len(pair_ids)}] {pair_id}/{role} -> {entry.latent_shape} in {entry.encode_seconds}s")

    print(f"wrote {n_done} latents under {out_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
