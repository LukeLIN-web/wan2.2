# Round-5 v3 lr1e5 warm lora_final PhyJudge eval (n=42)

**Verdict**: `fail`
**Criterion** (set in plan `humanize/dpo_v0/docs/exp-plan/round5_eval_criterion.md`): n=42 lora_final paired sign-test vs round-4 lora_final, p<0.05 favoring round-5.
**Next action**: revert canonical winner to round-4 β=100 lora_final; execute `humanize/dpo_v0/docs/eval/eval_v2_changeset.json` 7-step downstream per `eb4da377`.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_final.safetensors` (153,449,976 B) |
| ckpt sha256 | `4e8ca27979a9c710b89bc22b21ea3e5d001d55bed3d1f493106a815d6e771eff` |
| trainer step health | step 150/151 loss=1.5631 gnorm=16.3 margin=−1.33 (last single-pair raw) |
| eval gen out | `juyi-finetune:~/gen_out/round5_lora_final_n42_20260501T000726Z/` |
| baseline_sha256_match | true |
| n | 42 |
| baseline ref | `[ref:v3_baseline_n42]` (`~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`) |

## Headline (n=42)

### Paired sign-test (round-5 lora_final vs round-4 lora_final, criterion-binding)

| field | value |
|---|---|
| n_aligned | 42 |
| n_pos (round-5 better) | 4 |
| n_neg (round-4 better) | 19 |
| n_zero (tied) | 19 |
| p (two-sided binomial) | 0.0026 |
| reject H₀ at α=0.05 | true |
| direction | **favoring round-4** |
| **verdict** | **`fail`** |

> The test rejects the null hypothesis of equal performance with p=0.0026, but the direction of the effect is **favoring round-4 lora_final** (19 negative vs 4 positive per-prompt-axes-avg-Δ signs). Per the locked criterion (`pass` requires direction = favoring round-5), this is a `fail` from round-5's perspective. Round-4 β=100 lora_final remains the canonical winner.

### Round-5 lora_final vs baseline (informational, NOT verdict-binding)

| metric | Δ | 95% CI | sign-test p (vs 0) | vs criterion |
|---|---|---|---|---|
| axes-avg Δ | −0.038 | [−0.181, +0.090] | 1.000 | within-noise |

## Per-axis Δ (n=42, vs baseline)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | 0.000 | [−0.190, +0.190] | 1.000 |
| PTV | +0.048 | [−0.119, +0.214] | 0.774 |
| persistence | −0.143 | [−0.310, +0.024] | 0.227 |
| inertia | 0.000 | [−0.167, +0.167] | 1.000 |
| momentum | −0.095 | [−0.262, +0.048] | 0.508 |

All 5 axes within noise (every CI spans 0; every sign-test p > 0.05).

## Per-class Δ axes-avg (evalprompt.md A–G; n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 12 | +0.017 | [−0.150, +0.200] |
| B 破坏/形变 | 9 | −0.044 | [−0.400, +0.311] |
| C 流体 | 6 | +0.200 | [0.000, +0.400] |
| D 阴影/反射 | 5 | +0.080 | [0.000, +0.160] |
| E 链式 | 3 | −0.667 | n<5, no CI |
| F 滚动/滑动 | 4 | −0.200 | n<5, no CI |
| G 抛掷/弹道 | 2 | −0.100 | n<5, no CI |

> Bonferroni-adjusted α for 5 axes × 7 classes = 0.05/35 = 0.00143. None of the per-class CIs exclude 0 at this corrected level. C / D class CI lower bounds touch 0 at uncorrected α=0.05 (informational, not verdict-binding).

## Stuck-prompt watch (per-prompt scores, base ≤ 2)

| pid | class | base axes-mean | ckpt axes-mean | 5-axis (SA/PTV/persistence/inertia/momentum) |
|---|---|---|---|---|
| `2455740c4d45` | A | 2.0 | 1.4 | base 2/2/2/2/2 → ckpt 1/2/2/1/1 |

## Caveats (≤3)

- `evalprompt.md` G class lists 2 entries vs heading "n=3"; one G prompt is uncategorized → counted in global axes-avg only, not in per-class table. To fix in eval-v2 per `eb4da377` manifest.
- Bonferroni correction (5×7=35 tests) tightens α to 0.00143; per-class signals at uncorrected α=0.05 (C, D class CIs touching 0) do not survive correction. The verdict is bound only to the all-n=42 paired sign-test, not per-class.
- `lora_final` here is `step-151` (yaml-driven `num_samples=1202` ⇒ `target_steps=ceil(1202/8)=151`); 6 pair_ids see 1 epoch wrap-around (within DistributedSampler `set_epoch(1)` reuse).

## Source data

- aggregate JSON (canonical, stats.py): `juyi-finetune:/tmp/round5_lora_final_summary.json`
- paired sign-test JSON: `juyi-finetune:/tmp/paired_r5_vs_r4.json`
- round-5 trained scores (flat per-axis): `juyi-finetune:/tmp/round5_lora_final_trained_flat.jsonl`
- round-4 trained scores (flat per-axis, from #48 agg): `juyi-finetune:/tmp/round4_beta100_lora_final_trained_flat.jsonl`
- round-5 per-prompt scores: `juyi-finetune:~/gen_out/round5_lora_final_n42_20260501T000726Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos: `juyi-finetune:~/gen_out/round5_lora_final_n42_20260501T000726Z/20260501T000729Z/heldout_regen/<pid>/trained/...`
- baseline reuse: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
