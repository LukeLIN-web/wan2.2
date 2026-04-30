# Round-4 v3 lr1e5 — full 42-prompt validation (task #48)

**Task**: #48 (rl9). Full PhyJudge eval on 3 winning ckpts × all 42 heldout prompts (vs n=8/16 subsets in #46 rolling eval).

## TL;DR

| ckpt | n=42 axes-avg Δ vs baseline | rank |
|---|---|---|
| **β=100 lora_final (step-250)** | **+0.114** | **1 — WINNER** |
| β=100 step-150 | +0.000 | 2 |
| β=30 step-150 | −0.024 | 3 |

**β=100 lora_final wins at full 42-prompt scale.** All 5 axes positive (n=42 mean Δ): SA +0.143, PTV +0.190, persistence +0.024, inertia +0.167, momentum +0.048. β=30 step-150's n=8 lead (+0.200) was a small-sample artifact — at n=42 it's actually slightly *below* baseline.

## n=42 per-axis Δ (3 ckpts × 5 axes)

| axis | β=100 lora_final | β=30 step-150 | β=100 step-150 |
|---|---|---|---|
| **SA**          | **+0.143 ± 0.417** | +0.000 ± 0.698 | +0.024 ± 0.468 |
| **PTV**         | **+0.190 ± 0.455** | −0.024 ± 0.680 | +0.071 ± 0.513 |
| **persistence** | +0.024 ± 0.468 | −0.119 ± 0.705 | −0.095 ± 0.431 |
| **inertia**     | **+0.167 ± 0.660** | +0.048 ± 0.661 | +0.048 ± 0.582 |
| **momentum**    | +0.048 ± 0.492 | −0.024 ± 0.604 | −0.048 ± 0.539 |
| **axes-avg**    | **+0.114** ← winner | −0.024 | +0.000 |

β=100 lora_final wins on **all 5 axes** vs both other ckpts (only PTV-tie ties on β=100 step-150 inertia).

## Cross-cut: axes-avg Δ vs sample size (winner stability check)

| ckpt | n=8 (round-1/2 subset) | n=16 (round-3/4/5 subset) | n=42 (full validation) |
|---|---|---|---|
| **β=100 lora_final** | +0.200 | +0.062 | **+0.114** |
| β=30 step-150        | +0.200 | −0.037 | −0.024 |
| β=100 step-150       | +0.050 | +0.000 | +0.000 |

**Sample-size effect**:
- β=100 lora_final n=8 read of +0.200 was somewhat optimistic (regress to +0.114 at full 42), but **direction stable**.
- β=30 step-150 n=8 read of +0.200 was substantially over-optimistic — at n=42 the ckpt is actually net negative. The 8-prompt subset happened to be the prompts where β=30 step-150 happened to do well; the additional 34 prompts pull the mean down significantly.
- β=100 step-150 stable around 0 across all sample sizes.

**Lesson for round-5+**: n=8 single-step results have substantial subset-selection variance. Trajectory pattern (rolling step-50/100/150/200/250 at n=8) is still useful but absolute Δ values should be validated at larger n before declaring winners.

## Per-prompt-class breakdown at n=42

| ckpt | all (n=42) | non-collision (n=41) | collision (Newton's, n=1) |
|---|---|---|---|
| **β=100 lora_final** | **+0.114** | **+0.127** | −0.400 |
| β=30 step-150        | −0.024 | +0.000 | −1.000 |
| β=100 step-150       | +0.000 | +0.024 | −1.000 |

Even **excluding Newton's cradle**, β=100 lora_final still wins by +0.127 vs +0.000/+0.024 — its lead is not driven solely by collision-class recovery.

## Newton's cradle (`2455740c4d45`) at n=42

| axis | baseline | β=100 lora_final | β=30 step-150 | β=100 step-150 |
|---|---|---|---|---|
| SA          | 2 | **2** ⭐ | 1 | 1 |
| PTV         | 2 | **2** ⭐ | 1 | 1 |
| persistence | 2 | **2** ⭐ | 1 | 1 |
| inertia     | 2 | 1 | 1 | 1 |
| momentum    | 2 | 1 | 1 | 1 |

β=100 lora_final's 3/5 recovery on this prompt is unique among the 3 ckpts — confirmed at full 42-prompt validation (the same axes that recovered at the n=16 round-5 eval still recovered).

## Sign test: β=100 lora_final vs β=30 step-150 (n=42, paired per-prompt)

| axis | β=100 lora_final wins | β=30 step-150 wins | ties | sign-test p (two-sided) |
|---|---|---|---|---|
| SA          |  7 | 3 | 32 | 0.344 |
| **PTV**     | **10** | 2 | 30 | **0.039** ⭐ |
| persistence |  9 | 3 | 30 | 0.146 |
| inertia     |  7 | 3 | 32 | 0.344 |
| momentum    |  5 | 3 | 34 | 0.727 |

**Only PTV reaches statistical significance** (p=0.039) at the per-prompt level. Other axes directionally favor β=100 lora_final but don't reach p<0.05 in pairwise sign test. This is consistent with the small effect sizes (~0.1 per axis) and integer-quantized PhyJudge scale (most prompts tie because both ckpts give the same integer score).

The aggregate effect (+0.114 axes-avg Δ) is more reliable than any single-axis significance: across 5 × 42 = 210 axis-prompt pairs, β=100 lora_final wins on 38 vs β=30 step-150's 14, with 158 ties.

## Recommendation for round-5+

1. **Adopt β=100 lora_final (step-250) as the round-4 v3 lr1e5 winner ckpt.** Path: `videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234925Z/lora_final.safetensors` (153.4 MB, sha256 captured in earlier round mds).
2. **Statistical confidence is moderate** — PTV reaches p<0.05; other axes direction-of-travel only. The cross-axis aggregate (+0.114) is the more reliable signal.
3. **Newton's cradle gap remains** even at n=42: β=100 lora_final partially recovers (SA/PTV/persistence) but inertia/momentum stay floored. **Per-class fairness**: include collision-class breakdown in headline reporting going forward, not just axes-avg.
4. **Validate v3 reframe (`187189fa` U-shape forecast)**: at next high-β training (β≥100), keep `save_every=50` cadence so we capture trajectory; do NOT cut at step-200 cliff — recovery may follow at lora_final.
5. **Subset-selection variance is real**: future rolling eval should ideally use sliding subsets or random sub-samples rather than deterministic first-N to avoid biasing the n=8 read.

## Source data

- aggregate JSON: `juyi-finetune:~/gen_out/round4_lr1e5_full42_agg.json`
- per-ckpt scores: `juyi-finetune:~/gen_out/round4_lr1e5_beta{100,30}_step{150,250}_<ts>/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- baseline reuse: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`

## Methodology notes

- Resume idempotency on heldout_regen.py: required hot-patch (rl9) to honor existing `.run_ts` marker. Patched on both juyi-finetune and juyi-videorl. **Saved ~75 min wall** vs full-42 fresh regen on each ckpt by reusing the round-3/4/5 n=16 outputs (16 of 42 = 38% prompts pre-existing per ckpt).
- 3 ckpts split across 2 boxes: β=100 lora_final on juyi-finetune 4-rank (98 min), β=30 step-150 + β=100 step-150 chained on juyi-videorl 8-rank (56+56 min). Total wall ~113 min from re-launch (~2h actual including ship + score + agg + md).
- BitLesson candidates for round-5+ (heldout_regen.py + physground_score.py improvements):
  1. `--device cuda:${LOCAL_RANK}` per-rank (avoids 4-rank-all-cuda:0 OOM)
  2. `.run_ts` marker resume preserves run_dir for skip-if-complete idempotency
  3. physground_score.py per-video JSON keyed by `manifest.timestamp_utc` collides under rank parallel — use `--run-dir` per video into `scores_perprompt/<pid>/`
