# Round-6 plan — class-rebalanced DPO on eval-v2 (draft, pre-registration)

**Status**: DRAFT. Pre-registration window open until first round-6 ckpt eval is
aggregated. Per `howtoreport.md` §一 #5 ("决策门槛事前定，不事后改"), the
**Eval criterion** section MUST be frozen before any round-6 score is read.

This doc covers (a) training data sampling, (b) training config, (c)
pre-registered eval criterion, (d) decision tree. Mirrors round-5's
single-criterion file (`round5_eval_criterion.md`) but extended because the
round-6 novelty is on the data side, not the eval side.

## Background (why this round)

- Round-5 lora_final (warm-start from round-4 β=100 lora_final + 1202 setminus
  pairs) failed paired sign-test vs round-4 lora_final, p=0.0026 favoring
  round-4 (`round5_v3_lr1e5_warm_summary.md`).
- Diagnosis (`docs/data/distirbution.md`): round-4 1k and round-5 1202 are both
  drawn from the same A-heavy / B,D,E-light distribution as cond-present 2202.
  Per-class round-5 Δ regresses on E (−0.667), F (−0.200), G (−0.100); these
  three classes are the training-set's largest eval-vs-train shortfalls.
  Residual was random subsample, not low-margin filter — quality is comparable
  to round-4 1k; the failure is **distribution mismatch + same-distribution
  re-feed past saturation**, not pair quality.
- Eval-v2 is now in effect (`docs/eval/CHANGELOG.md`): A 12→15, E 3→1,
  total n=43 (manifest typo claims 42; CHANGELOG resolution: trust the explicit
  remove (2 E) + add (3 A) → 43).

## Hypothesis (single)

If round-6 trains on a **v2-eval-aligned 800-pair budget** sampled from the
2202 cond-present pool — A subsampled, B/D/E up-sampled — initialised from
round-4 β=100 lora_final, then it beats round-4 β=100 lora_final on the v2
n=43 paired sign-test. Everything else is held fixed (β=100, lr=1e-5,
lora_rank=16, num_samples=800 ⇒ target_steps=100 = 1 clean epoch on
world_size=8 / micro_batch=1, all other recipe knobs from
`training_config_round5_warm_beta100.yaml`).

## Training data sampling spec

### v2-aligned 800-pair budget

Eval-v2 class fractions (from `eval/PROMPT_CLASS.json` v2 / `eval_v2_changeset.json`,
n=43): A 15 / B 9 / C 6 / D 5 / E 1 / F 4 / G 3.

Budget = 800 (locked decision 5 below: `num_samples=800` ⇒ `target_steps =
ceil(800/8) = 100` = exactly 1 epoch on world_size=8 / micro_batch=1, no
wrap-around — cleaner than round-5's 6-pair wrap at num_samples=1202).

| class | eval v2 (n=43) | round-6 target (800 budget) | round-5 池 (cond-present 2202) | pool buffer | 操作 |
|---|---|---|---|---|---|
| A 多体碰撞 | 15 (34.9%) | **279** | 1131 | 852 | downsample 0.247× of pool |
| B 破坏/形变 | 9 (20.9%) | **167** | 214 | 47 | use ~78% of pool (decision 1: zero-buffer freeze, no disk-missing 回填) |
| C 流体 | 6 (14.0%) | **112** | 378 | 266 | downsample 0.296× |
| D 阴影/反射 | 5 (11.6%) | **93** | 134 | 41 | use ~69% of pool |
| E 链式 | 1 (2.3%) | **19** | 24 | 5 | use ~79% of pool (no repeat needed) |
| F 滚动 | 4 (9.3%) | **74** | 177 | 103 | downsample 0.418× |
| G 抛掷 | 3 (7.0%) | **56** | 113 | 57 | downsample 0.496× |
| **总计** | **43** | **800** | **2171** (excl. unclassified 31) | | |

Per-class quotas computed by floor(800 × eval_v2_fraction) with the
fractional remainders distributed by largest-remainder method to sum to
exactly 800 (A=279, B=167, C=112, D=93, E=19, F=74, G=56; sum=800). The
sampler must assert this distribution post-sample with `subset_pair_ids_sha256_hex16`
pin.

Eval-v1 distirbution.md L55–71 numbers (1k v1-aligned, E target 71 from
E 3/42) are superseded.

### Pool buffer at 800 budget — decision 1 implication

Decision 1 (locked): zero-buffer freeze on B's 167-of-214 selection — no
disk-missing 回填. At 800 budget B has 47-pair buffer (vs 5 at 1k budget)
so the freeze is workable. D / E similarly comfortable (41 / 5 buffer).
The disk-missing rescue path is left available for future rounds if needed.

### Sampler determinism

- Source: `humanize/dpo_v0/out/round4/20260428T160839Z/` cond-present 2202
  manifest (= round-2 raw 2745 setminus disk_missing 543, sha256-pinned via
  `T3_round4_tier_b_1k.json` provenance chain).
- Per-class oracle: `docs/data/distirbution.md` classifier rule (verified
  42/42 on heldout v1; v2 adds 3 A pids and removes 2 E pids — the rule
  re-applied to v2 is implied by the changeset and should be re-verified
  43/43 before sampling).
- Sampler script (TBD): `humanize/dpo_v0/script/sample/round6_class_balanced.py`.
  Produces `T3_round6_v2aligned_800.json` + `subset_pair_ids_sha256_hex16` pin.
  Seed namespace: `round6-v2aligned-tier_b-800-cond-present`.
- Output manifest must include per-class realized n (assert == target row in
  the table above) + sha256 pin.

## Training config spec

| field | value | rationale |
|---|---|---|
| init LoRA | round-4 β=100 lora_final (`ckpts/20260429T234925Z/lora_final.safetensors`) | canonical winner per CHANGELOG |
| init optim state | none (cold AdamW; round-4 didn't save optim) | round-7 will warm-resume from round-6's optim |
| `--save-optimizer-state` | **on** | F.11 contract; enables round-7 warm chain |
| β | 100 | round-4 證 β=100 > β=30 |
| lr | 1.0e-5 | matches round-4 / round-5 |
| lora_rank / lora_alpha | 16 / 16 | round-4 baseline |
| dpo_loss_kind | sigmoid | unchanged |
| sampling_band | [901, 999] | unchanged (high-noise 100% routing) |
| micro_batch | 1 | unchanged |
| num_samples | **800** | per decision 5 below; ceil(800/8)=100 → exactly 100 steps, no `--max-steps-override` patch needed, no epoch wrap-around |
| target_steps | **100** (derived from num_samples + world_size, no override) | round-5 step-100 is trajectory's only 健康点; 101→151 是末端 divergence; budget chosen to land exactly on it |
| save_every | **20** | 5 救生圈 ckpt (step-20/40/60/80/100); denser than round-4/5's 50 |
| seed_namespace | `round6-v2aligned-tier_b-800-cond-present` | |
| recipe yaml | `recipes/training_config_round6_v2aligned_beta100.yaml` (decision 3: fork from `training_config_round5_warm_beta100.yaml`) | |

Hard contracts (unchanged from round-4/5): total_forwards == num_samples ×
(target_steps × micro_batch × world_size / num_samples) per F.11; routing
100% high-noise; no halt / no OOM; baseline_sha256_match: true at eval time.

## Eval criterion (pre-registered, locked)

> **Round-6 lora_final n=43 verdict = `pass`** iff paired sign-test
> (round-6 lora_final, round-4 β=100 lora_final) on **identical 43 v2
> prompt-ids**, per-prompt-axes-avg-Δ sign vs 0 (two-sided binomial,
> α=0.05), with the test direction set to **favoring round-6**.
>
> Else `verdict = fail` (canonical winner stays round-4 lora_final;
> round-6 ckpt is not adopted).

### Operational definitions

- **Identical 43 v2 prompt-ids**: as defined in `eval/PROMPT_CLASS.json` v2
  (post-`eb4da377` manifest land); 40 reused from eval-v1 r4 lora_final scores
  + 3 newly-evaluated A pids (`366d2a1252b3` / `5b7bb71f101d` / `d8b29a78eed7`)
  per CHANGELOG L70–74.
- **Per-prompt-axes-avg-Δ**: `Δ_p = mean_axes(round_6[p][axis]) - mean_axes(round_4[p][axis])`,
  5 PhyJudge axes equal weight (SA / PTV / persistence / inertia / momentum).
- **Sign-test**: `scipy.stats.binomtest`, `alternative="two-sided"`, p=0.5,
  α=0.05.
- **Direction "favoring round-6"**: required for `pass`. Two-sided p<0.05 with
  `n_pos > n_neg`. p<0.05 with `n_neg > n_pos` = `fail`.

### Stats implementation

`humanize/dpo_v0/eval/stats.py` (rl8 owner). No hand-math in result md.

- `paired_sign_test(deltas_round6, deltas_round4_v2, alpha=0.05) -> {"p", "n_pos", "n_neg", "n_tie", "verdict"}`
- `bootstrap_ci(deltas, n_resamples=10000, alpha=0.05, resample_unit="prompt") -> (lo, hi)`
- `sign_test_vs_zero(...)` for the supplementary baseline read.
- Bonferroni: `α_corrected = 0.05 / (5 * 7)` for the per-class supplementary
  table (informational only — not verdict-binding).

### Inputs (pinned at criterion-freeze time)

| field | source |
|---|---|
| round-6 lora_final scores | `juyi-finetune:~/gen_out/round6_lora_final_n43_<TS>/scores_perprompt/<pid>/<run_ts>/results.jsonl` |
| round-4 β=100 lora_final v2 scores (40 reused + 3 new) | merged from `juyi-finetune:~/gen_out/round4_lr1e5_beta100_step250_<TS>/` (40 v1-shared pids) + `juyi-finetune:~/gen_out/eval_v2_baseline_plus_r4_n3_20260501T031745Z/` (3 new A pids) |
| baseline videos (v2) | regen of v3 baseline on the 3 new A pids + reuse of `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z` for 40 shared |
| baseline scores (v2) | merged: 3 new from `eval_v2_baseline_plus_r4_n3_<TS>/scores/` + 40 from `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl` |

`baseline_sha256_match: true` asserted at runtime per eval pipeline; failure
voids the round-6 result md (`howtoreport.md` §一 #10).

### What this criterion does NOT include (intentionally)

- No axes-avg point-estimate threshold (no "≥ +0.114" cutoff). Cross-n / cross-set
  point comparisons are forbidden by `howtoreport.md` §一 #2.
- No per-class verdict-binding test. The 5×7=35 per-class slice goes in the
  result md as a descriptive supplement with Bonferroni note, not as a gate.
- No early-ckpt verdict. step-{20,40,60,80} eval are `rolling-read-only` per
  `howtoreport.md` §一 #1; only lora_final is verdict-binding.
- No Newton's-cradle / single-prompt verdict. Stuck-prompt watch table only.

## Decision tree

| outcome | action |
|---|---|
| `pass` (p<0.05 favoring round-6) | adopt round-6 lora_final as canonical winner; commit `lora_final_optim.pt` for round-7 warm-resume; freeze v2 r6 reference numbers in CHANGELOG |
| `draw` (p ≥ 0.05) | round-4 stays canonical. Diagnose: was the data plan insufficient (suggests need for B/D pool回填 + re-run), or is the round-4 saturation point harder than data alone can break (suggests next round explores fresh-init from base, not warm) |
| `fail` (p<0.05 favoring round-4) | round-4 stays canonical. Same diagnosis as draw, more strongly weighted toward "warm-from-r4 is past saturation; fresh-init needed for round-7" |

In all three branches, `lora_final_optim.pt` is preserved so that any
subsequent round can choose between (i) warm-from-round-6 or (ii) warm-from-round-4
with optim state (the latter requires a separate round-4 re-run with
`--save-optimizer-state` on, since the original round-4 didn't save optim).

## Decisions (locked 2026-04-30 by user)

1. **B 类回填策略 = (a) zero-buffer freeze**. No disk-missing 回填. Pool 214,
   target 167 (at 800 budget), 47-pair buffer is workable. The disk-missing
   rescue path stays available for future rounds if needed.
2. **v2 classifier re-verify = rerun**. Apply the rule code at
   `docs/data/distirbution.md` L83–156 to v2's `eval/PROMPT_CLASS.json` (43
   pids) before sampling; assert 43/43 match against the v2 manifest's
   class-of-record. Block the sampler run on this assertion.
3. **Recipe yaml = fork**. Create
   `recipes/training_config_round6_v2aligned_beta100.yaml` from
   `training_config_round5_warm_beta100.yaml`. Swaps:
   `num_samples: 1202 → 800`, `max_pairs: 1202 → 800`,
   `seed_namespace: round5-warm-tier_b-1202-cond-present →
   round6-v2aligned-tier_b-800-cond-present`,
   `round_tag: round-5-warm → round-6-v2aligned`,
   `subset_pair_ids_sha256_hex16: 680a7eec8090d48b → <new pin>` (filled in
   after sampler runs). Pin sha256_hex16 of the new yaml. Per
   `feedback_no_comments.md`: no narrative header / change-log comments in
   the yaml.
4. **Eval-v2 r4 lora_final n=43 reference = 合并**. Merge:
   - 40 v1-shared pids: reuse from `juyi-finetune:~/gen_out/round4_lr1e5_beta100_step250_<TS>/scores_perprompt/`
   - 3 new A pids: from `juyi-finetune:~/gen_out/eval_v2_baseline_plus_r4_n3_20260501T031745Z/scores/`
   - Output: single jsonl `round4_beta100_lora_final_v2_n43_flat.jsonl` with
     sha256 pin. Owner: rl8 / eval-infra. Must produce + pin before criterion
     freeze.
5. **target_steps cap = pre-truncate `num_samples` to 800 in the recipe**. No
   trainer patch. With `num_samples=800`, world_size=8, micro_batch=1, the
   default formula `target_steps = ceil(800/8) = 100` lands exactly on the
   cap. Bonus: 100 × 8 = 800 pairs/epoch = exactly 1 epoch, zero wrap-around
   (round-5's 6-pair wrap at num_samples=1202 is gone).

## Source data pinning

- Eval-v2 manifest (frozen pre-result): `docs/eval/eval_v2_changeset.json`
  (luke1 `4b28c95`)
- Eval-v2 changelog: `docs/eval/CHANGELOG.md` (entry "eval-v2 effective
  2026-05-01")
- Class distribution analysis (eval-v1; needs v2 update): `docs/data/distirbution.md`
- Round-5 fail summary: `docs/experiment-results/round5_v3_lr1e5_warm_summary.md`
- Round-4 winner reference (eval-v1, now historical): `docs/experiment-results/round4_v3_lr1e5_summary.md`
  + `docs/experiment-results/round4_v3_lr1e5_full42_validation.md`
- Round-5 recipe (round-6 forks from this): `recipes/training_config_round5_warm_beta100.yaml`
- Pool manifest source: `humanize/dpo_v0/out/round4/20260428T160839Z/T3_round4_tier_b_1k.json`
  (and the 2202 cond-present derivation chain)

## Authoring + change control

- **Author**: TBD (draft for owner pickup)
- **Pre-registration freeze**: this doc must be committed at HEAD before any
  round-6 lora_final score is aggregated. Post-result modification = process
  violation per `howtoreport.md` §一 #5.
- **Change-control after freeze**: any modification requires `--allow-empty`
  commit recording (1) timestamp of result aggregation, (2) proposer, (3)
  approver, (4) reason.

## Sign-off

- TBD — pending owner assignment for: training plan author, criterion author,
  stats.py owner endorsement, coordinator endorsement, luke1 directive.
