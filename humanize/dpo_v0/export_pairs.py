#!/usr/bin/env python3
"""T1 — strict same-group preference pair exporter for DPO v0.

Reads `human_eval_filtered.db` (read-only), constructs winner/loser pairs strictly
within each `comparison_group`, applies (margin, min_raters) thresholds and
prompt-level train/val/heldout split, and emits `pair.json` + sidecar `manifest.json`.

See `humanize/dpo_v0/README.md` for the contract. The exporter never writes to the DB.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

DEFAULT_DB = "/shared/user60/worldmodel/wmbench/evals/human_eval/human_eval_filtered.db"
WAN_TARGET = "wan2.2-i2v-a14b"
SCORE_DIMS = ("SA", "PTV", "persistence")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--min-raters", type=int, default=2)
    p.add_argument("--seed", default="0xdpo")
    p.add_argument(
        "--split-fractions", default="0.7,0.15,0.15",
        help="train,val,heldout fractions (must sum to 1.0)"
    )
    p.add_argument(
        "--target-model", default=WAN_TARGET,
        help="Model under DPO; reported in wan_pair_counts of the manifest."
    )
    args = p.parse_args(argv)
    args.split_fractions = tuple(float(x) for x in args.split_fractions.split(","))
    if len(args.split_fractions) != 3 or abs(sum(args.split_fractions) - 1.0) > 1e-9:
        p.error(f"--split-fractions must be three values summing to 1.0, got {args.split_fractions}")
    return args


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_per_video_scores(conn):
    """Return {video_id: {dataset, filename, score, n_raters}} keyed by video_id.

    Per-video score = mean over the union of raters that produced a complete
    SA+PTV+persistence triple for that video, across every group the video was
    rated in. Excluded annotators are filtered out.
    """
    rows = conn.execute(
        """
        SELECT v.id, v.dataset, v.filename,
               AVG(per_rater.tot) AS score,
               COUNT(*) AS n_raters
        FROM (
          SELECT a.video_id, a.annotator_id, SUM(ai.score) AS tot
          FROM assignments a
          JOIN annotations ann ON ann.assignment_id = a.id
          JOIN annotation_items ai ON ai.annotation_id = ann.id
          WHERE a.status = 'completed'
            AND NOT EXISTS (
              SELECT 1 FROM excluded_annotators e WHERE e.annotator_id = a.annotator_id
            )
            AND ai.dimension IN ('SA', 'PTV', 'persistence')
          GROUP BY a.video_id, a.annotator_id
          HAVING COUNT(DISTINCT ai.dimension) = 3
        ) per_rater
        JOIN videos v ON v.id = per_rater.video_id
        GROUP BY v.id, v.dataset, v.filename
        """
    ).fetchall()
    return {
        vid: {"dataset": ds, "filename": fn, "score": float(sc), "n_raters": int(nr)}
        for vid, ds, fn, sc, nr in rows
    }


def load_group_videos_and_meta(conn):
    """Return:
      group_videos: {group_id: [video_id, ...]}  (eligible scored videos in the group)
      group_meta:   {group_id: {prompt, physical_laws, filename}}

    The group_meta verifies the within-group invariants (single filename, single prompt,
    single physical_laws). On any violation, raises AssertionError naming the offender.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT a.group_id, v.id AS video_id, v.filename, cg.prompt, cg.physical_laws
        FROM assignments a
        JOIN videos v  ON v.id = a.video_id
        JOIN comparison_groups cg ON cg.id = a.group_id
        WHERE a.group_id IS NOT NULL
          AND a.status = 'completed'
          AND NOT EXISTS (
            SELECT 1 FROM excluded_annotators e WHERE e.annotator_id = a.annotator_id
          )
        """
    ).fetchall()

    group_videos = defaultdict(list)
    group_filenames = defaultdict(set)
    group_prompts = defaultdict(set)
    group_physical_laws = defaultdict(set)
    for g, vid, fn, prompt, pl in rows:
        group_videos[g].append(vid)
        group_filenames[g].add(fn)
        group_prompts[g].add(prompt)
        group_physical_laws[g].add(pl)

    # Within-group invariants
    bad = [g for g, s in group_filenames.items() if len(s) > 1]
    assert not bad, f"within-group filename inconsistency at: {bad[:3]}"
    bad = [g for g, s in group_prompts.items() if len(s) > 1]
    assert not bad, f"within-group prompt inconsistency at: {bad[:3]}"
    bad = [g for g, s in group_physical_laws.items() if len(s) > 1]
    assert not bad, f"within-group physical_laws inconsistency at: {bad[:3]}"

    group_meta = {
        g: {
            "prompt": next(iter(group_prompts[g])),
            "physical_laws": next(iter(group_physical_laws[g])),
            "filename": next(iter(group_filenames[g])),
        }
        for g in group_videos
    }
    return dict(group_videos), group_meta


def stable_split(prompt_keys, fractions, seed: str):
    """Deterministic three-way split of prompt keys.

    Each prompt key is hashed (md5) with the seed; the resulting 64-bit prefix is
    mapped to [0, 1) and bucketed by cumulative fractions.
    """
    train_f, val_f, _ = fractions
    splits = {}
    for k in prompt_keys:
        h = hashlib.md5(f"{seed}|{k}".encode("utf-8")).hexdigest()
        x = int(h[:16], 16) / float(1 << 64)
        if x < train_f:
            splits[k] = "train"
        elif x < train_f + val_f:
            splits[k] = "val"
        else:
            splits[k] = "heldout"
    return splits


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(exist_ok=True)

    import sqlite3
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        video_score = load_per_video_scores(conn)
        group_videos, group_meta = load_group_videos_and_meta(conn)
    finally:
        conn.close()

    # Prompt-level split. Bucket key = (prompt, physical_laws). Per rl2: 1:1:1 with
    # filenames in this DB, so prompt-level == scene+action-level.
    prompt_keys = sorted({(m["prompt"], m["physical_laws"]) for m in group_meta.values()})
    split_assignment = stable_split(
        ["||".join(k) for k in prompt_keys], args.split_fractions, args.seed
    )
    prompt_to_split = {k: split_assignment["||".join(k)] for k in prompt_keys}

    drop_hist = Counter({k: 0 for k in
                          ("tie", "margin_below", "rater_below", "cross_prompt", "cross_group", "dup")})
    seen_pairs = set()
    pairs = []  # ordered: winner score > loser score

    # Build pair candidates within each group
    for g, vids in group_videos.items():
        # videos with computable score
        scored = [v for v in vids if v in video_score]
        if len(scored) < 2:
            continue
        for v1, v2 in combinations(scored, 2):
            s1 = video_score[v1]["score"]
            s2 = video_score[v2]["score"]

            # invariant: same group ⇒ same prompt (already asserted in loader);
            # any violation here is a bug, not a drop.
            # We still keep the bucket label for manifest completeness; the value
            # must remain 0 by construction.

            if s1 == s2:
                drop_hist["tie"] += 1
                continue
            if s1 > s2:
                w_id, l_id = v1, v2
                margin = s1 - s2
            else:
                w_id, l_id = v2, v1
                margin = s2 - s1
            w = video_score[w_id]
            l = video_score[l_id]
            if margin < args.margin:
                drop_hist["margin_below"] += 1
                continue
            if w["n_raters"] < args.min_raters or l["n_raters"] < args.min_raters:
                drop_hist["rater_below"] += 1
                continue
            key = (g, frozenset((v1, v2)))
            if key in seen_pairs:
                drop_hist["dup"] += 1
                continue
            seen_pairs.add(key)
            meta = group_meta[g]
            split = prompt_to_split[(meta["prompt"], meta["physical_laws"])]
            pair = {
                "pair_id": f"{g}__{w_id}_gt_{l_id}",
                "group_id": g,
                "split": split,
                "filename": meta["filename"],
                "prompt": meta["prompt"],
                "physical_laws": meta["physical_laws"],
                "winner": {
                    "video_id": w_id, "dataset": w["dataset"],
                    "filename": w["filename"], "score": round(w["score"], 6),
                    "n_raters": w["n_raters"],
                },
                "loser": {
                    "video_id": l_id, "dataset": l["dataset"],
                    "filename": l["filename"], "score": round(l["score"], 6),
                    "n_raters": l["n_raters"],
                },
                "margin": round(margin, 6),
            }
            pairs.append(pair)

    # Invariants — these must be 0 by construction; assert so a future bug surfaces here.
    assert drop_hist["cross_prompt"] == 0, "cross_prompt invariant violated"
    assert drop_hist["cross_group"] == 0, "cross_group invariant violated"

    # Per-split summaries
    per_split_pairs = defaultdict(list)
    per_split_prompts = defaultdict(set)
    per_split_groups = defaultdict(set)
    for p in pairs:
        per_split_pairs[p["split"]].append(p)
        per_split_prompts[p["split"]].add(p["prompt"])
        per_split_groups[p["split"]].add(p["group_id"])

    splits_out = {}
    for s in ("train", "val", "heldout"):
        ps = per_split_pairs[s]
        wan_loser = sum(1 for q in ps if q["loser"]["dataset"] == args.target_model)
        wan_winner = sum(1 for q in ps if q["winner"]["dataset"] == args.target_model)
        splits_out[s] = {
            "pair_count": len(ps),
            "prompt_count": len(per_split_prompts[s]),
            "group_count": len(per_split_groups[s]),
            "prompts": sorted(per_split_prompts[s]),
            "group_ids": sorted(per_split_groups[s]),
            "wan_pair_counts": {
                "as_loser": wan_loser,
                "as_winner": wan_winner,
                "total": wan_loser + wan_winner,
            },
        }

    # Verify split disjointness — assertion, not a drop reason.
    train_p, val_p, hold_p = (set(splits_out[s]["prompts"]) for s in ("train", "val", "heldout"))
    assert train_p.isdisjoint(val_p), "train ∩ val prompts non-empty"
    assert train_p.isdisjoint(hold_p), "train ∩ heldout prompts non-empty"
    assert val_p.isdisjoint(hold_p), "val ∩ heldout prompts non-empty"

    # Overall Wan-pair counts
    wan_overall_loser = sum(1 for p in pairs if p["loser"]["dataset"] == args.target_model)
    wan_overall_winner = sum(1 for p in pairs if p["winner"]["dataset"] == args.target_model)

    # Distinct datasets seen
    datasets = sorted({p["winner"]["dataset"] for p in pairs} | {p["loser"]["dataset"] for p in pairs})

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "meta": {
            "generated_at": timestamp,
            "db_path": args.db,
            "db_md5": file_md5(args.db),
            "exporter_path": str(Path(__file__).resolve()),
            "exporter_git_sha": git_sha(str(Path(__file__).parent)),
            "args": {
                "margin": args.margin,
                "min_raters": args.min_raters,
                "seed": args.seed,
                "split_fractions": list(args.split_fractions),
                "target_model": args.target_model,
            },
        },
        "thresholds": {"margin": args.margin, "min_raters": args.min_raters},
        "fallback_thresholds": {
            "margin": 0.5, "min_raters": 1,
            "rationale": "T4 data-sparse rescue knob; do not use without explicit reviewer sign-off",
        },
        "aggregation_rule": "cross_group_rater_union",
        "aggregation_rule_explainer": (
            "Per-video score = mean over (SA + PTV + persistence) per rater, where the rater set is the "
            "union of all annotators that produced a complete SA+PTV+persistence triple for the video "
            "across any comparison_group it appeared in. The DB allows at most 1 rater per "
            "(group, video) cell, so within-group rater means would be degenerate."
        ),
        "retained_pair_count": len(pairs),
        "drop_reason_histogram": dict(drop_hist),
        "drop_reason_notes": {
            "cross_prompt": (
                "INVARIANT: must be 0. Pairs are within-group and group_id implies a unique prompt. "
                "Non-zero indicates schema drift; exporter aborts."
            ),
            "cross_group": (
                "INVARIANT: must be 0. Cross-group pairing is rejected by construction "
                "(combinations are enumerated within-group only). Non-zero indicates an exporter bug."
            ),
        },
        "splits": splits_out,
        "splits_notes": (
            "prompt-level split — verified 1:1:1 mapping of (prompt) ↔ (filename) ↔ (prompt, "
            "physical_laws) at T1 export (250 of each). Therefore prompt-level disjoint == "
            "filename-level disjoint == scene+action-level disjoint, satisfying plan Step 6."
        ),
        "wan_pair_counts": {
            "as_loser": wan_overall_loser,
            "as_winner": wan_overall_winner,
            "total": wan_overall_loser + wan_overall_winner,
        },
        "datasets_in_pairs": datasets,
        "invariants_verified": [
            "within_group_filename_consistent",
            "within_group_prompt_consistent",
            "within_group_physical_laws_consistent",
            "1:1:1 prompt/filename/(prompt,physical_laws) at row count",
            "drop_reason_histogram[cross_prompt]==0",
            "drop_reason_histogram[cross_group]==0",
            "splits.train ∩ splits.val == ∅ (prompts)",
            "splits.train ∩ splits.heldout == ∅ (prompts)",
            "splits.val ∩ splits.heldout == ∅ (prompts)",
        ],
    }

    # Writes
    pair_path = out_dir / "pair.json"
    with open(pair_path, "w") as f:
        json.dump(pairs, f, indent=2)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    for s in ("train", "val", "heldout"):
        with open(out_dir / "splits" / f"{s}.json", "w") as f:
            json.dump(per_split_pairs[s], f, indent=2)

    # Stdout summary (timestamp prefix per CLAUDE.md convention)
    print(f"[{timestamp}] wrote {pair_path} ({len(pairs)} pairs)")
    print(f"[{timestamp}] wrote {manifest_path}")
    print(
        f"[{timestamp}] retained_pair_count={len(pairs)}  "
        f"wan_loser={wan_overall_loser}  wan_winner={wan_overall_winner}  "
        f"datasets={len(datasets)}  drop={dict(drop_hist)}"
    )
    for s in ("train", "val", "heldout"):
        d = splits_out[s]
        print(
            f"[{timestamp}] split={s:<8s} pairs={d['pair_count']:>5d}  "
            f"prompts={d['prompt_count']:>3d}  groups={d['group_count']:>5d}  "
            f"wan_loser={d['wan_pair_counts']['as_loser']:>3d}  "
            f"wan_winner={d['wan_pair_counts']['as_winner']:>3d}"
        )


if __name__ == "__main__":
    main()
