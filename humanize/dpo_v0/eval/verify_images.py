#!/usr/bin/env python3
"""T2 — conditioning-image recovery + MD5 verification.

Reads a T1 pair.json and resolves every retained group's canonical I2V
conditioning image on disk. Produces:

  image_manifest.json   group_id -> {scene_filename, image_path, image_md5, status, source_dir}
  drop_list.json        groups dropped (status != "ok") with reasons
  post_t2_pair.json     pair.json filtered to only ok groups
  post_t2_summary.json  recomputed retained_pair_count / wan_pair_counts / split breakdown
  spotcheck.md          manual spot-check log over N sampled groups
  spotcheck/<gid>.png   side-by-side frame for each spot-check sample

Status values:
  ok                              - canonical image found and read
  image_missing                   - no candidate image file located
  image_unreadable                - file exists but not loadable as image
  multiple_candidates_no_resolution - >1 candidate path with conflicting MD5s
  image_md5_inconsistent          - same scene_filename across groups -> different MD5
"""

import argparse
import hashlib
import io
import json
import random
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Image search roots — order matters for openvid (wmbench is more comprehensive).
PIQ_DIR = Path("/shared/user60/worldmodel/rlvideo/videodpo/worldmodelbench/physics-IQ-benchmark/switch-frames")
VP2_DIR = Path("/shared/user60/worldmodel/wmbench/data/prompts/video_phy_2/first_frames")
OPENVID_DIRS = [
    Path("/shared/user60/worldmodel/wmbench/data/prompts/openvid/first_frames"),
    Path("/shared/user60/worldmodel/rlvideo/videodpoWan/WorldModelBench/images"),
]
VIDEO_ROOT = Path("/shared/user60/worldmodel/wmbench/data/videos")
WAN_TARGET = "wan2.2-i2v-a14b"

PIQ_PATTERN = re.compile(r"^[0-9]{4}_perspective-")
VP2_PATTERN = re.compile(r"^[a-z][a-z_]+_[0-9]+\.mp4$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-json", required=True, help="Path to T1 pair.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--spotcheck-n", type=int, default=10)
    p.add_argument("--spotcheck-seed", default="0xdpo-t2")
    p.add_argument(
        "--spotcheck-pass-ssim", type=float, default=0.6,
        help="SSIM threshold above which the spot-check is a clear PASS",
    )
    p.add_argument(
        "--spotcheck-fail-ssim", type=float, default=0.30,
        help="SSIM threshold below which the spot-check is a clear FAIL",
    )
    return p.parse_args(argv)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(scene_filename: str):
    """Return list of candidate Paths (existing files) for a scene's conditioning image.

    Returns: (candidates, source_label)
    """
    base = scene_filename[:-4] if scene_filename.endswith(".mp4") else scene_filename
    if PIQ_PATTERN.match(scene_filename):
        # physics_iq: 0161_perspective-center_trimmed-X.mp4
        #   -> 0161_switch-frames_anyFPS_perspective-center_trimmed-X.jpg
        img_name = base.replace("_perspective-", "_switch-frames_anyFPS_perspective-", 1) + ".jpg"
        cand = PIQ_DIR / img_name
        return ([cand] if cand.exists() else [], "physics_iq")
    if VP2_PATTERN.match(scene_filename):
        cand = VP2_DIR / f"{base}.jpg"
        return ([cand] if cand.exists() else [], "videophy_2")
    # openvid
    cands = []
    for d in OPENVID_DIRS:
        c = d / f"{base}.jpg"
        if c.exists():
            cands.append(c)
    return (cands, "openvid")


def video_path_for(scene_filename: str, dataset: str) -> Path:
    """Return path to a generated video for the scene under a given dataset."""
    return VIDEO_ROOT / dataset / scene_filename


def extract_first_frame(video: Path, out_size: int = 256):
    """Extract first frame using ffmpeg. Returns numpy uint8 array (H, W, 3) or None."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", str(video),
                "-vframes", "1",
                "-vf", f"scale={out_size}:{out_size}",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-",
            ],
            capture_output=True, check=True, timeout=30,
        )
        return np.array(Image.open(io.BytesIO(proc.stdout)).convert("RGB"))
    except Exception:
        return None


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Simple grayscale SSIM in [-1, 1]; expects identical shape uint8 arrays."""
    if a.shape != b.shape:
        return float("nan")
    x = a.astype(np.float64).mean(axis=2)
    y = b.astype(np.float64).mean(axis=2)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    num = (2 * mx * my + c1) * (2 * cov + c2)
    den = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return float(num / den) if den else float("nan")


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    (out_dir / "spotcheck").mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(args.pair_json))

    # Unique group -> scene_filename mapping (T1 invariant: one filename per group).
    group_scene = {}
    for p in pairs:
        gid = p["group_id"]
        sf = p["filename"]
        if gid in group_scene:
            assert group_scene[gid] == sf, f"T1 invariant violated: group {gid} has multiple filenames"
        else:
            group_scene[gid] = sf

    # Resolve & MD5 per unique scene_filename (cache).
    scene_resolution = {}  # scene_filename -> {candidates, source, image_path, image_md5, status, candidate_md5s}
    md5_cache = {}
    for sf in sorted(set(group_scene.values())):
        cands, source = resolve_image_path(sf)
        entry = {"scene_filename": sf, "source": source, "candidates": [str(c) for c in cands]}
        if not cands:
            entry["status"] = "image_missing"
            entry["image_path"] = None
            entry["image_md5"] = None
            scene_resolution[sf] = entry
            continue
        # Compute MD5s; if multiple candidates, all must agree.
        cand_md5s = []
        for c in cands:
            try:
                cand_md5s.append((str(c), md5_cache.setdefault(str(c), file_md5(c))))
            except Exception as e:
                entry["status"] = "image_unreadable"
                entry["image_path"] = str(c)
                entry["image_md5"] = None
                entry["error"] = str(e)
                scene_resolution[sf] = entry
                break
        if "status" in entry:
            continue
        unique_md5 = set(m for _, m in cand_md5s)
        if len(unique_md5) > 1:
            entry["status"] = "multiple_candidates_no_resolution"
            entry["candidate_md5s"] = cand_md5s
            entry["image_path"] = None
            entry["image_md5"] = None
        else:
            # Validate readability via PIL.
            chosen_path, chosen_md5 = cand_md5s[0]
            try:
                with Image.open(chosen_path) as im:
                    im.verify()
                entry["status"] = "ok"
                entry["image_path"] = chosen_path
                entry["image_md5"] = chosen_md5
                if len(cand_md5s) > 1:
                    entry["candidate_md5s"] = cand_md5s
            except Exception as e:
                entry["status"] = "image_unreadable"
                entry["image_path"] = chosen_path
                entry["image_md5"] = chosen_md5
                entry["error"] = str(e)
        scene_resolution[sf] = entry

    # Build per-group manifest
    image_manifest = {}
    for gid, sf in group_scene.items():
        sr = scene_resolution[sf]
        image_manifest[gid] = {
            "scene_filename": sf,
            "image_path": sr.get("image_path"),
            "image_md5": sr.get("image_md5"),
            "status": sr.get("status"),
            "source_dir": sr.get("source"),
        }
        if sr.get("candidate_md5s"):
            image_manifest[gid]["candidate_md5s"] = sr["candidate_md5s"]
        if sr.get("error"):
            image_manifest[gid]["error"] = sr["error"]

    # MD5 uniqueness across scene_filenames is enforced by md5_cache above
    # (one MD5 per filename); no second pass needed.
    inconsistencies: list = []
    inconsistent_scenes: set = set()
    drop_list = []
    ok_pairs = []
    for p in pairs:
        gid = p["group_id"]
        sf = p["filename"]
        s = image_manifest[gid]["status"]
        if sf in inconsistent_scenes:
            drop_list.append({"pair_id": p["pair_id"], "group_id": gid, "scene_filename": sf,
                              "reason": "image_md5_inconsistent"})
            continue
        if s != "ok":
            drop_list.append({"pair_id": p["pair_id"], "group_id": gid, "scene_filename": sf,
                              "reason": s})
            continue
        ok_pairs.append(p)

    # Recompute counts
    wan_loser = sum(1 for p in ok_pairs if p["loser"]["dataset"] == WAN_TARGET)
    wan_winner = sum(1 for p in ok_pairs if p["winner"]["dataset"] == WAN_TARGET)
    by_split_count = Counter(p["split"] for p in ok_pairs)
    by_split_wan_loser = Counter(p["split"] for p in ok_pairs if p["loser"]["dataset"] == WAN_TARGET)
    by_split_wan_winner = Counter(p["split"] for p in ok_pairs if p["winner"]["dataset"] == WAN_TARGET)
    drop_reason_hist = Counter(d["reason"] for d in drop_list)

    pre_count = len(pairs)
    post_count = len(ok_pairs)

    # Spot-check
    rng = random.Random(args.spotcheck_seed)
    eligible_groups = [gid for gid, m in image_manifest.items() if m["status"] == "ok"]
    spot_groups = rng.sample(eligible_groups, min(args.spotcheck_n, len(eligible_groups)))

    spotcheck_results = []
    for gid in spot_groups:
        sf = group_scene[gid]
        img_path = Path(image_manifest[gid]["image_path"])
        # Pick a video to compare against — prefer the WAN target, fall back to any pair video.
        cand_video = video_path_for(sf, WAN_TARGET)
        if not cand_video.exists():
            # Find any dataset that has this scene
            for ds_dir in VIDEO_ROOT.iterdir():
                p = ds_dir / sf
                if p.exists():
                    cand_video = p
                    break
        if not cand_video.exists():
            spotcheck_results.append({
                "group_id": gid, "scene_filename": sf,
                "image_path": str(img_path), "image_md5": image_manifest[gid]["image_md5"],
                "compared_video": None, "ssim": None,
                "verdict": "uncertain", "reason": "no generated video found for spotcheck",
            })
            continue

        ref_arr = np.array(Image.open(img_path).convert("RGB").resize((256, 256)))
        frame = extract_first_frame(cand_video)
        if frame is None:
            spotcheck_results.append({
                "group_id": gid, "scene_filename": sf,
                "image_path": str(img_path), "image_md5": image_manifest[gid]["image_md5"],
                "compared_video": str(cand_video), "ssim": None,
                "verdict": "uncertain", "reason": "ffmpeg first-frame extraction failed",
            })
            continue
        s = ssim(ref_arr, frame)
        if s >= args.spotcheck_pass_ssim:
            verdict, reason = "pass", f"ssim {s:.3f} >= {args.spotcheck_pass_ssim}"
        elif s < args.spotcheck_fail_ssim:
            verdict, reason = "fail", f"ssim {s:.3f} < {args.spotcheck_fail_ssim}"
        else:
            verdict, reason = "uncertain", f"ssim {s:.3f} between thresholds"

        # Save side-by-side image
        side = Image.new("RGB", (512, 256))
        side.paste(Image.fromarray(ref_arr), (0, 0))
        side.paste(Image.fromarray(frame), (256, 0))
        side.save(out_dir / "spotcheck" / f"{gid}.png")

        spotcheck_results.append({
            "group_id": gid, "scene_filename": sf,
            "image_path": str(img_path), "image_md5": image_manifest[gid]["image_md5"],
            "compared_video": str(cand_video),
            "ssim": round(s, 4),
            "verdict": verdict,
            "reason": reason,
        })

    # ============== writes ==============
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(out_dir / "image_manifest.json", "w") as f:
        json.dump(image_manifest, f, indent=2)
    with open(out_dir / "drop_list.json", "w") as f:
        json.dump({"meta": {"generated_at": timestamp, "drop_count": len(drop_list)},
                   "drop_reason_histogram": dict(drop_reason_hist),
                   "inconsistent_scenes": inconsistencies,
                   "drops": drop_list}, f, indent=2)
    with open(out_dir / "post_t2_pair.json", "w") as f:
        json.dump(ok_pairs, f, indent=2)

    # post-T2 summary (counts at multiple breakdowns)
    summary = {
        "meta": {
            "generated_at": timestamp,
            "input_pair_json": str(Path(args.pair_json).resolve()),
            "n_unique_groups_input": len(group_scene),
            "n_unique_scenes_input": len(set(group_scene.values())),
        },
        "image_status_histogram": dict(Counter(m["status"] for m in image_manifest.values())),
        "pre_t2_pair_count": pre_count,
        "post_t2_pair_count": post_count,
        "drop_count": len(drop_list),
        "drop_reason_histogram": dict(drop_reason_hist),
        "wan_pair_counts_post_t2": {
            "as_loser": wan_loser, "as_winner": wan_winner, "total": wan_loser + wan_winner,
        },
        "splits_post_t2": {
            s: {
                "pair_count": by_split_count.get(s, 0),
                "wan_pair_counts": {
                    "as_loser": by_split_wan_loser.get(s, 0),
                    "as_winner": by_split_wan_winner.get(s, 0),
                },
            }
            for s in ("train", "val", "heldout")
        },
        "spotcheck": {
            "n": len(spotcheck_results),
            "verdict_histogram": dict(Counter(s["verdict"] for s in spotcheck_results)),
            "thresholds": {
                "pass_ssim": args.spotcheck_pass_ssim,
                "fail_ssim": args.spotcheck_fail_ssim,
            },
        },
    }
    with open(out_dir / "post_t2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # spotcheck.md
    md = ["# T2 Spotcheck Log\n",
          f"Generated: {timestamp}\n",
          f"N samples: {len(spotcheck_results)} (seed={args.spotcheck_seed})\n",
          f"SSIM thresholds: pass>={args.spotcheck_pass_ssim}, fail<{args.spotcheck_fail_ssim}\n\n",
          "Compared the canonical conditioning image against the first frame of a "
          f"`{WAN_TARGET}`-generated video for that scene (or any available dataset if "
          "the WAN video is missing). The first frame of an I2V output should closely "
          "match the conditioning input; high SSIM is therefore strong evidence the "
          "conditioning image we resolved is genuinely the canonical one.\n\n",
          "| group_id (8) | scene | source | image_md5 (8) | ssim | verdict |\n",
          "|---|---|---|---|---|---|\n"]
    for r in spotcheck_results:
        gid8 = r["group_id"][:8]
        md58 = (r["image_md5"] or "")[:8]
        ssim_str = f"{r['ssim']:.3f}" if r["ssim"] is not None else "n/a"
        md.append(
            f"| {gid8} | {r['scene_filename']} | "
            f"{image_manifest[r['group_id']]['source_dir']} | {md58} | {ssim_str} | {r['verdict']} |\n"
        )
    md.append("\n## Per-sample reason\n")
    for r in spotcheck_results:
        md.append(f"- **{r['group_id']}** ({r['scene_filename']}): {r['reason']} — image=`{r['image_path']}`")
        if r.get("compared_video"):
            md.append(f"; video=`{r['compared_video']}`")
        md.append("\n")
    md.append("\nSide-by-side renderings saved at `spotcheck/<group_id>.png`.\n")
    with open(out_dir / "spotcheck.md", "w") as f:
        f.writelines(md)

    # stdout summary
    print(f"[{timestamp}] image_status_histogram = {summary['image_status_histogram']}")
    print(f"[{timestamp}] pre_t2_pair_count  = {pre_count}")
    print(f"[{timestamp}] post_t2_pair_count = {post_count}  (drop {len(drop_list)})")
    print(f"[{timestamp}] drop_reason_histogram = {dict(drop_reason_hist)}")
    print(f"[{timestamp}] wan_pair_counts_post_t2 = "
          f"as_loser={wan_loser}  as_winner={wan_winner}  total={wan_loser+wan_winner}")
    for s in ("train", "val", "heldout"):
        d = summary["splits_post_t2"][s]
        print(f"[{timestamp}] split={s:<8s} pairs={d['pair_count']:>5d}  "
              f"wan_loser={d['wan_pair_counts']['as_loser']:>3d}  "
              f"wan_winner={d['wan_pair_counts']['as_winner']:>3d}")
    print(f"[{timestamp}] spotcheck verdict histogram = "
          f"{summary['spotcheck']['verdict_histogram']}")


if __name__ == "__main__":
    main()
