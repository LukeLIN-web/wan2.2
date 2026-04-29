#!/usr/bin/env python3
"""Round-4 task #19: build the 1k-pair tier_b subset for lr=5e-5/200-step proof-of-method.

Inheritance from round-2:
- T3 subset (tier_b 2745 pair_ids) reused unchanged.
- T2 image_manifest reused unchanged (status='ok' for all 1195 group_ids in tier_b).
- Heldout 42-scene set reused unchanged (prompt-disjoint constraint preserved).

Round-3 lesson (M4 200 -> 160 effective):
- 200 pairs ingested but 40 silently filtered at trainer dataset construction
  due to cond_image disk-missing (image_manifest status was 'ok' but file
  not actually on disk for some physics-IQ-benchmark scenes).
- Round-4 pre-filter at subset-build time so the trainer sees exactly N pairs.

Selection:
- Filter tier_b 2745 -> 2202 (cond_image disk-present, scene-disjoint heldout).
- Deterministic shuffle: seed = sha256("round4-tier_b-1k" || recipe_id)[:8] hex.
- Take first 1000.

Output:
  humanize/dpo_v0/out/round4/<UTC>/T3_round4_tier_b_1k.json
  humanize/dpo_v0/out/round4/<UTC>/recipe_id_pin   (frozen recipe_id at build time)

The file is gitignored. Commit only the script + a marker pointing at the UTC dir.
"""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--t3-subset-json", required=True,
                   help="Path to round-2 T3_subset.json (tier_b pair_ids source)")
    p.add_argument("--pair-json", required=True,
                   help="Path to round-2 pair.json (pair_id -> {group_id, filename, prompt, ...})")
    p.add_argument("--t2-image-manifest", required=True,
                   help="Path to round-2 t2/image_manifest.json")
    p.add_argument("--heldout-json", required=True,
                   help="Path to round-2 splits/heldout.json (scene-disjoint check)")
    p.add_argument("--recipe-id", required=True,
                   help="Frozen recipe_id pin (sha256[:16] of canonical recipe YAML)")
    p.add_argument("--target-n", type=int, default=1000,
                   help="Number of pairs to select")
    p.add_argument("--out-dir", required=True,
                   help="Output dir (a UTC subdir will be created)")
    p.add_argument("--seed-namespace", default="round4-tier_b-1k",
                   help="Seed namespace string for deterministic shuffle")
    return p.parse_args(argv)


def load_inputs(args):
    t3 = json.loads(pathlib.Path(args.t3_subset_json).read_bytes())
    pair_list = json.loads(pathlib.Path(args.pair_json).read_bytes())
    image_manifest = json.loads(pathlib.Path(args.t2_image_manifest).read_bytes())
    heldout = json.loads(pathlib.Path(args.heldout_json).read_bytes())
    return t3, pair_list, image_manifest, heldout


def filter_present(tier_b_ids, pair_by_id, image_manifest, heldout_scenes):
    drops = {
        "missing_in_pair_json": [],
        "group_missing_in_image_manifest": [],
        "image_status_not_ok": [],
        "image_path_disk_missing": [],
        "scene_in_heldout": [],
    }
    accepted = []
    for pid in sorted(tier_b_ids):
        pair = pair_by_id.get(pid)
        if pair is None:
            drops["missing_in_pair_json"].append(pid)
            continue
        gid = pair["group_id"]
        im_entry = image_manifest.get(gid)
        if im_entry is None:
            drops["group_missing_in_image_manifest"].append(pid)
            continue
        if im_entry.get("status") != "ok":
            drops["image_status_not_ok"].append({"pair_id": pid, "status": im_entry.get("status")})
            continue
        image_path = im_entry["image_path"]
        if not pathlib.Path(image_path).is_file():
            drops["image_path_disk_missing"].append({"pair_id": pid, "image_path": image_path})
            continue
        if pair["filename"] in heldout_scenes:
            drops["scene_in_heldout"].append({"pair_id": pid, "scene": pair["filename"]})
            continue
        accepted.append(pid)
    return accepted, drops


def compute_seed(namespace: str, recipe_id: str) -> str:
    payload = f"{namespace}||{recipe_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def canonical_pair_ids_sha256(pair_ids: list[str]) -> str:
    """Canonical hash for the trainer pin (rl2 review #19 follow-up).

    Newline-joined + trailing newline form, NOT json.dumps default form, so it
    is independent of CPython json.dumps default separators which may vary
    across implementations / future versions.
    """
    payload = ("\n".join(pair_ids) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv=None):
    args = parse_args(argv)

    t3, pair_list, image_manifest, heldout = load_inputs(args)
    tier_b_ids = list(t3["tier_b"]["pair_ids"])
    pair_by_id = {p["pair_id"]: p for p in pair_list}
    heldout_scenes = set(p["filename"] for p in heldout)

    accepted, drops = filter_present(tier_b_ids, pair_by_id, image_manifest, heldout_scenes)
    if len(accepted) < args.target_n:
        sys.stderr.write(
            f"ERROR: accepted={len(accepted)} < target_n={args.target_n}; "
            f"halt to avoid silent under-sampling.\n"
        )
        sys.exit(2)

    seed_hex = compute_seed(args.seed_namespace, args.recipe_id)
    seed_int = int(seed_hex, 16)
    rng = random.Random(seed_int)
    shuffled = list(accepted)
    rng.shuffle(shuffled)
    selected = shuffled[: args.target_n]

    selected_records = [pair_by_id[pid] for pid in selected]
    selected_groups = {r["group_id"] for r in selected_records}
    selected_scenes = {r["filename"] for r in selected_records}

    leak = sorted(selected_scenes & heldout_scenes)
    if leak:
        sys.stderr.write(f"ERROR: scene leakage with heldout: {leak}\n")
        sys.exit(3)

    pair_ids_sha256_full = canonical_pair_ids_sha256(selected)
    pair_ids_sha256_hex16 = pair_ids_sha256_full[:16]

    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.out_dir) / utc
    out_dir.mkdir(parents=True, exist_ok=True)

    out_payload = {
        "meta": {
            "task": "round4-task-19",
            "tier_b_source_subset": str(args.t3_subset_json),
            "tier_b_source_pair_count": len(tier_b_ids),
            "filter": {
                "cond_image_present_pair_count": len(accepted),
                "dropped_total": sum(len(v) for v in drops.values()),
                "drop_reason_counts": {k: len(v) for k, v in drops.items()},
                "sample_disk_missing_paths": [
                    d["image_path"] for d in drops["image_path_disk_missing"][:5]
                ],
            },
            "selection": {
                "target_n": args.target_n,
                "selected_count": len(selected),
                "seed_namespace": args.seed_namespace,
                "recipe_id_pin": args.recipe_id,
                "seed_hex8": seed_hex,
                "seed_int": seed_int,
                "method": "sha256(namespace||recipe_id)[:8] -> Random.shuffle -> take first N",
            },
            "pair_ids_sha256_canonical": {
                "form": "newline-joined-with-trailing-newline",
                "code": "hashlib.sha256(('\\n'.join(pair_ids) + '\\n').encode('utf-8')).hexdigest()",
                "sha256_full": pair_ids_sha256_full,
                "sha256_hex16": pair_ids_sha256_hex16,
            },
            "stats": {
                "n_unique_groups": len(selected_groups),
                "n_unique_scenes": len(selected_scenes),
            },
            "utc": utc,
        },
        "tier_b_round4_1k": {
            "pair_ids": selected,
            "group_ids": sorted(selected_groups),
            "scene_filenames": sorted(selected_scenes),
        },
        "leakage_check": {
            "selected_scenes_intersect_heldout": leak,
            "heldout_scene_count": len(heldout_scenes),
        },
    }

    out_path = out_dir / "T3_round4_tier_b_1k.json"
    out_path.write_bytes(
        (json.dumps(out_payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")
    )

    pin_path = out_dir / "recipe_id_pin"
    pin_path.write_text(args.recipe_id + "\n", encoding="utf-8")

    pids_pin_path = out_dir / "pair_ids_sha256_hex16_pin"
    pids_pin_path.write_text(pair_ids_sha256_hex16 + "\n", encoding="utf-8")

    drops_path = out_dir / "drop_log.json"
    drops_path.write_bytes(
        (json.dumps({"counts": {k: len(v) for k, v in drops.items()}, "details": drops},
                    ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )

    print(f"OK selected {len(selected)} pairs from {len(accepted)} accepted (of {len(tier_b_ids)} tier_b source).")
    print(f"  unique_groups={len(selected_groups)} unique_scenes={len(selected_scenes)}")
    print(f"  seed={seed_hex} (int={seed_int}) recipe_id={args.recipe_id}")
    print(f"  leakage_with_heldout={len(leak)}")
    print(f"  out_dir={out_dir}")
    print(f"  out_path={out_path}")


if __name__ == "__main__":
    main()
