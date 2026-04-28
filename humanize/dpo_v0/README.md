# DPO v0 — strict same-group preference pair exporter (T1)

Exports the strict same-group preference pairs needed for the v0 minimal DPO loop on
`Wan2.2-I2V-A14B_high_noise_full_round2`.

## Rationale

A `comparison_group` in `human_eval_filtered.db` is identified by a unique
`(prompt, physical_laws)` tuple. Every video in a group shares the same `filename`
(the source/scene+action id) and only the producing model (`videos.dataset`) varies.
Per rl2's verification: prompts ↔ filenames ↔ (prompt, physical_laws) is a perfect
1:1:1 mapping (250 of each). A pair-export that respects `group_id` therefore
implicitly satisfies the plan's "same prompt + action + scene + conditioning_image"
constraint. Pair-export rejects cross-group pairing as an invariant assertion, not
a normal drop reason.

## Per-video score

Plan T0 wording: *per-video score = SA + PTV + persistence, mean over the video's
available raters*. The DB has at most 1 rater per `(group, video)` cell because
`assignments(video_id, annotator_id)` is unique and each `(video, annotator)` pair
appears in only one group. The non-trivial reading is therefore: **mean over the
union of raters that scored the video across all groups it appears in**. We call
this the "cross-group rater union" aggregation rule and stamp it in the manifest.

## Outputs

Outputs land under `humanize/dpo_v0/out/<timestamp>/` and are gitignored
(`*.json` is in `.gitignore` for this repo). The exporter is reproducible from the
DB given the same script, args, and seed.

```
out/<timestamp>/
├── pair.json                   # full retained pair list with split labels
├── manifest.json               # sidecar with thresholds, drop histogram, splits, invariants
├── splits/train.json           # subset of pair.json filtered to split=train
├── splits/val.json
└── splits/heldout.json
```

## Default thresholds

`margin >= 1.0`, `n_raters >= 2`. Override via `--margin` / `--min-raters`. The
fallback knob `(0.5, 1)` is documented for T4 data-sparse rescue (see manifest).

## Splits

`train / val / heldout` are computed at the **prompt level**: the unique
`(prompt, physical_laws)` strings are deterministically hashed (with `--seed`) and
bucketed by `--split-fractions`. All groups for a prompt land in the same split.
Because of the 1:1:1 invariant, prompt-level disjoint == filename-level disjoint
== scene+action-level disjoint, satisfying the plan Step 6 requirement.

## Drop-reason histogram

Six buckets in the manifest:

- `tie` — winner.score == loser.score (dropped).
- `margin_below` — `|score_w - score_l| < margin`.
- `rater_below` — winner.n_raters < R or loser.n_raters < R.
- `cross_prompt` — **invariant assertion, must be 0**. If non-zero, the DB schema
  has shifted under us (a single `group_id` would imply more than one prompt) and
  the exporter aborts.
- `cross_group` — **invariant assertion, must be 0**. Pairs are constructed
  strictly within `group_id`. If non-zero, the exporter has a bug.
- `dup` — symmetric pair `(group_id, {v1, v2})` already emitted.

## Reproducibility

```
python3 humanize/dpo_v0/export_pairs.py \
    --db /shared/user60/worldmodel/wmbench/evals/human_eval/human_eval_filtered.db \
    --out-dir humanize/dpo_v0/out/$(date -u +%Y%m%dT%H%M%SZ) \
    --margin 1.0 --min-raters 2 --seed 0xdpo \
    --split-fractions 0.7,0.15,0.15
```

DB MD5 is recorded in `manifest.json[meta][db_md5]`. Exporter git SHA recorded in
`manifest.json[meta][exporter_git_sha]`.
