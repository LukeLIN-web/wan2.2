# Round-5 lora_final eval — pre-registered criterion (frozen)

**Pre-registered before round-3 result write per `howtoreport.md` §一 #5
("决策门槛事前定，不事后改").**

This document locks the round-5 lora_final winner-vs-round-4-lora_final test
**before** round-3 (n=42) eval results are aggregated. Any change after this
file is committed = process violation.

## Criterion (verbatim, locked at commit time)

> **round-5 lora_final n=42 verdict = `pass`** iff paired sign-test
> (round-5 lora_final, round-4 lora_final) on **identical 42 prompt-ids**,
> per-prompt-axes-avg-Δ sign vs 0 (two-sided binomial, α=0.05),
> with the test direction set to **favoring round-5**.
>
> Else `verdict = fail` (canonical winner reverts to round-4 lora_final).

## Operational definitions

- **Identical 42 prompt-ids**: the heldout set as defined in
  `<T0_T3_ROOT>/splits/heldout.json`, same canonical `prompt_id =
  sha256(prompt)[:12]` form used in round-4 #46/#48.
- **Per-prompt-axes-avg-Δ**: for each prompt p,
  `Δ_p = mean_axes(round_5_score[p][axis]) - mean_axes(round_4_score[p][axis])`,
  averaged across the 5 PhyJudge axes (SA, PTV, persistence, inertia,
  momentum) with equal weight.
- **Sign vs 0**: per-prompt sign of Δ_p (positive favors round-5, negative
  favors round-4, zero = tie).
- **Two-sided binomial sign-test**: standard `scipy.stats.binomtest`
  (or equivalent) over the 42 signs, `alternative="two-sided"`, `p=0.5`,
  α=0.05.
- **Direction "favoring round-5"**: required for `pass`. Two-sided p<0.05 with
  more positive-sign prompts than negative-sign prompts. p<0.05 with more
  negative signs = `fail` (round-4 wins, but reported as `fail` from round-5's
  perspective).

## Statistics implementation

All numbers computed via `humanize/dpo_v0/eval/stats.py` (rl8 owner).
**No hand-math in the round-3 md.** Functions used:

- `paired_sign_test(deltas_round5, deltas_round4, alpha=0.05) -> {"p": float, "n_pos": int, "n_neg": int, "n_tie": int, "verdict": "pass"|"fail"}`
- `bootstrap_ci(deltas, n_resamples=10000, alpha=0.05, resample_unit="prompt") -> (lo, hi)`
- `sign_test_vs_zero(deltas, alpha=0.05) -> {"p": float, "verdict_within_noise": bool}`
- Bonferroni: `α_corrected = 0.05 / (5 * num_classes)` for multi-class breakdown.

## Inputs to the test (data-source pinning)

| field | source |
|---|---|
| round-5 lora_final scores | `juyi-finetune:~/gen_out/round5_lora_final_n42_<TS>/scores_perprompt/<pid>/<run_ts>/results.jsonl` |
| round-4 lora_final scores (β=100, n=42) | `juyi-finetune:~/gen_out/round4_lr1e5_beta100_step250_<TS>/scores_perprompt/<pid>/<run_ts>/results.jsonl` (from #48 full-42 validation) |
| baseline videos | `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z` (reused via `--baseline-from`) |
| baseline scores | `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl` (reused) |

`baseline_sha256_match: true` is asserted at runtime by the eval pipeline; if
it ever fails, the round-3 md is voided per `howtoreport.md` §一 #10.

## What this criterion does NOT include (intentionally)

- **No axes-avg point-estimate threshold** (no "round-5 ≥ +0.114" cutoff).
  Cross-n point comparisons are explicitly forbidden by `howtoreport.md` §一
  #2.
- **No PTV-axis sign-test** as a separate verdict (PTV was post-hoc-elevated
  in round-4 #48; per `howtoreport.md` §一 #5 it does not count as a
  pre-registered criterion for round-5).
- **No "Newton's cradle 5/5 vs 3/5"** verdict — n=1 prompt does not constitute
  evidence per `howtoreport.md` §一 #3. Newton's cradle goes in the
  stuck-prompt watch table only.
- **No multi-stage gating** — the sign-test on n=42 axes-avg-Δ is the single
  binding test. If it passes, ckpt ships. If it fails, revert.

## Per-class supplementary (informational, not verdict-binding)

Per `howtoreport.md` §二 + `evalprompt.md` A-G classes, the round-3 md will
include per-class axes-avg Δ + 95% CI for n≥5 classes (A=12, B=9, C=6, D=5,
F=4, G=3, E=3 — last 3 raw-Δ-only). These are **descriptive supplements**,
not verdict criteria. The verdict is bound only to the all-n=42 paired
sign-test above.

## Authoring + change control

- **Author**: rl9
- **Frozen-at-commit**: yes
- **Change-control**: any modification to this criterion must be committed
  with `--allow-empty` + a note explicitly recording:
  1. timestamp of round-3 result aggregation
  2. who proposed the change
  3. who approved
  4. why
  After round-3 result write, this doc is read-only. Per `howtoreport.md` §一
  #5 a post-hoc change is a process violation.

## Sign-off

- **rl9** (criterion author, plan-doc owner)
- **rl2** (coordinator endorse, msg `ff485e61`)
- **rl8** (stats.py author, msg `de9b7c69` — proposed identical wording)
- **luke1** (directive `8c4279ee` 设定 howtoreport.md authority; this
  criterion conforms to §一 + §四)
