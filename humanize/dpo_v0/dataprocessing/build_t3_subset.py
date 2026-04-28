#!/usr/bin/env python3
"""Build the T3 pre-encode subset list and dump it to T3_subset.json.

Pure selection logic — no encoding, no VAE, no GPU. Reads the T2-filtered
post_t2_pair.json and splits/{train,val}.json. Writes:
  humanize/dpo_v0/T3_subset.json:
    tier_a              16 pair_ids (T4 tiny-overfit, stratified by Wan role)
    tier_b              all train+val pair_ids (T5 short DPO encode subset)
    heldout_excluded    all heldout group_ids and scenes (anti-leakage proof)

The heldout_excluded list is what rl2 will grep against tier_a/tier_b to verify
no overlap.
"""

import argparse
import json
import random
from pathlib import Path


WAN_TARGET = "wan2.2-i2v-a14b"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--post-t2-pair-json", required=True)
    p.add_argument("--train-json", required=True)
    p.add_argument("--val-json", required=True)
    p.add_argument("--heldout-json", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--tier-a-size", type=int, default=16)
    p.add_argument("--tier-a-seed", default="0xdpo-t4")
    return p.parse_args(argv)


def select_tier_a(train_pairs, n: int, seed: str):
    """Stratified pick: half wan-as-loser, half wan-as-winner.

    Within each stratum, prefer diverse opponent models (round-robin by opponent
    dataset); break ties with deterministic random order.
    """
    rng = random.Random(seed)
    half = n // 2

    wan_loser = [p for p in train_pairs if p["loser"]["dataset"] == WAN_TARGET]
    wan_winner = [p for p in train_pairs if p["winner"]["dataset"] == WAN_TARGET]
    rng.shuffle(wan_loser)
    rng.shuffle(wan_winner)

    def diverse_take(pool, k, opp_key):
        # Greedy round-robin: at each step pick the pair whose opponent dataset
        # has the lowest cumulative pick count, breaking ties by pair_id.
        seen_opp = {p[opp_key]["dataset"]: 0 for p in pool}
        out = []
        remaining = list(pool)
        while len(out) < k and remaining:
            remaining.sort(key=lambda p: (seen_opp[p[opp_key]["dataset"]], p["pair_id"]))
            chosen = remaining.pop(0)
            out.append(chosen)
            seen_opp[chosen[opp_key]["dataset"]] += 1
        return out

    a_loser = diverse_take(wan_loser, half, "winner")     # diverse opponents = winner side
    a_winner = diverse_take(wan_winner, n - half, "loser") # diverse opponents = loser side
    return a_loser + a_winner


def main(argv=None):
    args = parse_args(argv)

    train_pairs = json.loads(Path(args.train_json).read_bytes())
    val_pairs = json.loads(Path(args.val_json).read_bytes())
    heldout_pairs = json.loads(Path(args.heldout_json).read_bytes())
    post_t2 = json.loads(Path(args.post_t2_pair_json).read_bytes())

    # Tier A: 16 stratified pair_ids from train.
    tier_a = select_tier_a(train_pairs, n=args.tier_a_size, seed=args.tier_a_seed)
    tier_a_ids = [p["pair_id"] for p in tier_a]
    tier_a_videos = sorted({(p["winner"]["video_id"], p["winner"]["dataset"]) for p in tier_a}
                            | {(p["loser"]["video_id"], p["loser"]["dataset"]) for p in tier_a})
    tier_a_scenes = sorted({p["filename"] for p in tier_a})

    # Tier B: all train+val pair_ids (post-T2). Constructed from the sorted union
    # to be deterministic regardless of input file order.
    train_val_pairs = sorted(train_pairs + val_pairs, key=lambda p: p["pair_id"])
    tier_b_ids = [p["pair_id"] for p in train_val_pairs]
    tier_b_videos = sorted({(p["winner"]["video_id"], p["winner"]["dataset"]) for p in train_val_pairs}
                            | {(p["loser"]["video_id"], p["loser"]["dataset"]) for p in train_val_pairs})
    tier_b_scenes = sorted({p["filename"] for p in train_val_pairs})

    heldout_pair_ids = sorted(p["pair_id"] for p in heldout_pairs)
    heldout_scenes = sorted({p["filename"] for p in heldout_pairs})
    heldout_groups = sorted({p["group_id"] for p in heldout_pairs})
    heldout_videos = sorted({(p["winner"]["video_id"], p["winner"]["dataset"]) for p in heldout_pairs}
                              | {(p["loser"]["video_id"], p["loser"]["dataset"]) for p in heldout_pairs})

    # Anti-leakage cross-check: assert no scene overlap.
    overlap_a_h = sorted(set(tier_a_scenes) & set(heldout_scenes))
    overlap_b_h = sorted(set(tier_b_scenes) & set(heldout_scenes))
    assert not overlap_a_h, f"tier_a leaks into heldout: {overlap_a_h}"
    assert not overlap_b_h, f"tier_b leaks into heldout: {overlap_b_h}"

    # Sanity: all tier ids must exist in post_t2.
    post_t2_ids = {p["pair_id"] for p in post_t2}
    missing = [pid for pid in tier_a_ids + tier_b_ids if pid not in post_t2_ids]
    assert not missing, f"subset references unknown pair_ids: {missing[:3]}"

    out = {
        "meta": {
            "post_t2_pair_count": len(post_t2),
            "train_pair_count": len(train_pairs),
            "val_pair_count": len(val_pairs),
            "heldout_pair_count": len(heldout_pairs),
        },
        "tier_a": {
            "name": "T4 tiny-overfit",
            "size": len(tier_a_ids),
            "seed": args.tier_a_seed,
            "stratification": "8 wan-as-loser + 8 wan-as-winner; diverse opponents",
            "pair_ids": tier_a_ids,
            "unique_videos": [{"video_id": v, "dataset": d} for v, d in tier_a_videos],
            "unique_scene_filenames": tier_a_scenes,
        },
        "tier_b": {
            "name": "T5 short DPO subset",
            "size": len(tier_b_ids),
            "composition": "all train + val pair_ids (post-T2)",
            "n_unique_videos": len(tier_b_videos),
            "n_unique_scenes": len(tier_b_scenes),
            "pair_ids": tier_b_ids,
            "unique_videos": [{"video_id": v, "dataset": d} for v, d in tier_b_videos],
            "unique_scene_filenames": tier_b_scenes,
        },
        "heldout_excluded": {
            "name": "NEVER pre-encode (anti-leakage)",
            "rationale": (
                "Heldout videos are reserved for plan Step 6 regeneration with both "
                "round2 baseline and trained ckpt under shared generation_config. "
                "Pre-encoding them would leak the human-judged target into training."
            ),
            "pair_ids": heldout_pair_ids,
            "group_ids": heldout_groups,
            "scene_filenames": heldout_scenes,
            "n_unique_videos": len(heldout_videos),
        },
        "leakage_check": {
            "tier_a_scenes ∩ heldout_scenes": [],
            "tier_b_scenes ∩ heldout_scenes": [],
            "tier_a_ids ⊂ post_t2": True,
            "tier_b_ids ⊂ post_t2": True,
        },
    }

    Path(args.out_path).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out_path}")
    print(f"tier_a: {out['tier_a']['size']} pairs / "
          f"{len(out['tier_a']['unique_videos'])} unique videos / "
          f"{len(tier_a_scenes)} scenes")
    print(f"tier_b: {out['tier_b']['size']} pairs / "
          f"{out['tier_b']['n_unique_videos']} unique videos / "
          f"{out['tier_b']['n_unique_scenes']} scenes")
    print(f"heldout_excluded: {len(heldout_pair_ids)} pairs / "
          f"{len(heldout_groups)} groups / {len(heldout_scenes)} scenes")
    print(f"leakage check (scenes): tier_a∩heldout=0  tier_b∩heldout=0  ✓")


if __name__ == "__main__":
    main()
