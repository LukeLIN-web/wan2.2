#!/usr/bin/env python
"""Merge per-GPU v2 evaluation shards into a single results file."""
import argparse
import json
import numpy as np
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="Shard file prefix (e.g. run_dir/shard)")
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--output", required=True, help="Output JSON path")
    args = p.parse_args()

    merged_preds = {}
    merged_accs = defaultdict(list)
    model_name = ""

    for gpu_id in range(args.num_gpus):
        for suffix in ("_cot.json", ".json"):
            path = f"{args.prefix}_gpu{gpu_id}{suffix}"
            if Path(path).exists():
                break
        else:
            print(f"Warning: no shard for GPU {gpu_id}")
            continue

        with open(path) as f:
            shard = json.load(f)

        if not model_name:
            model_name = shard.get("model_name", "")
        merged_preds.update(shard.get("preds", {}))
        for k, v in shard.get("accs", {}).items():
            merged_accs[k].extend(v)

    num_videos = len(merged_preds)
    scores = {}
    if "instruction" in merged_accs:
        scores["instruction_following"] = float(np.mean(merged_accs["instruction"]))

    # Extract interleaved sub-category scores for multi-question dimensions
    dimension_subs = {
        "physical_laws": ("physics", ["Newton", "Mass", "Fluid", "Penetration", "Gravity"]),
        "common_sense": ("cs", ["Aesthetics", "Temporal"]),
    }
    for dim_key, (prefix, sub_names) in dimension_subs.items():
        if dim_key not in merged_accs:
            continue
        values = merged_accs[dim_key]
        num_subs = len(sub_names)
        sub_means = []
        for j, name in enumerate(sub_names):
            sub = [values[k] for k in range(j, len(values), num_subs)]
            m = float(np.mean(sub))
            scores[f"{prefix}_{name.lower()}"] = m
            sub_means.append(m)
        scores[dim_key] = float(np.mean(sub_means))

    # Overall = weighted mean (1 instr + 5 physics + 2 cs = 8 judgments per video)
    all_vals = (
        merged_accs.get("instruction", [])
        + merged_accs.get("physical_laws", [])
        + merged_accs.get("common_sense", [])
    )
    scores["overall"] = float(np.mean(all_vals)) if all_vals else 0.0

    result = {
        "model_name": model_name,
        "version": "v2",
        "num_videos": num_videos,
        "scores": scores,
        "counts": {k: len(v) for k, v in merged_accs.items()},
        "preds": merged_preds,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Merged {num_videos} videos")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
