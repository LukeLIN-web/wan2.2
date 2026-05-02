# Round-5 v3 lr1e5 warm step-100 PhyJudge eval (n=24)

**Verdict**: `rolling-read-only`
**Criterion** (set in plan `humanize/dpo_v0/docs/exp-plan/round5_eval_criterion.md`): n=42 lora_final paired sign-test vs round-4 lora_final, p<0.05 favoring round-5.
**Next action**: no decisions taken at this checkpoint.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_step100.safetensors` (153,449,976 B) |
| ckpt sha256 | `9280fa5aefa1e7eba439f1a45f347a8be5a88332fbb092b98e4d1bd2446a443f` |
| trainer step health | loss=0.5681, gnorm=10.4, margin=+0.268, vram_peak_alloc=58.35GB |
| eval gen out | `juyi-finetune:~/gen_out/round5_step100_n24_20260430T220301Z/` |
| baseline_sha256_match | true |
| n | 24 |
| baseline ref | `[ref:v3_baseline_n42]` (`~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`) |

## Headline (n=24)

| metric | Δ | 95% CI | sign-test p (vs 0) | vs criterion |
|---|---|---|---|---|
| axes-avg Δ | −0.008 | [−0.167, +0.158] | 1.000 | within-noise |
| A 多体碰撞 (n=8) | +0.075 | [−0.100, +0.250] | — | within-noise |
| 其他类合计 (n=16) | (n_per_class<5, no CI) | — | — | rolling-read-only |

## Per-axis Δ (n=24)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | +0.042 | [−0.125, +0.208] | 1.000 |
| PTV | −0.042 | [−0.250, +0.167] | 1.000 |
| persistence | −0.042 | [−0.250, +0.167] | 1.000 |
| inertia | 0.000 | [−0.208, +0.250] | 1.000 |
| momentum | 0.000 | [−0.292, +0.250] | 1.000 |

All 5 axes within noise.

## Per-class Δ axes-avg (evalprompt.md A–G; n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 8 | +0.075 | [−0.100, +0.250] |
| B 破坏/形变 | 4 | −0.100 | n<5, no CI |
| C 流体 | 3 | +0.267 | n<5, no CI |
| D 阴影/反射 | 4 | +0.050 | n<5, no CI |
| E 链式 | 1 | −0.200 | n<5, no CI |
| F 滚动/滑动 | 3 | −0.200 | n<5, no CI |
| G 抛掷/弹道 | 1 | −0.600 | n<5, no CI |

## Stuck-prompt watch (per-prompt scores, base ≤ 2)

| pid | class | base axes-mean | ckpt axes-mean | 5-axis (SA/PTV/persistence/inertia/momentum) |
|---|---|---|---|---|
| `2455740c4d45` | A | 2.0 | 1.6 | base 2/2/2/2/2 → ckpt 2/2/2/1/1 |

## Caveats (≤3)

- n=24 < 42 prompts → `rolling-read-only`. All Δ readouts here are direction-of-travel only and do not constitute evidence per `howtoreport.md` §一 #1.
- Only A class has n≥5 in this rolling subset; B/C/D/E/F/G show raw Δ only.
- Round-4 lora_final reference [ref:r4_lora_final_n42] is on n=42 prompts; cross-n point-estimate comparison is forbidden per `howtoreport.md` §一 #2 and is intentionally not shown.

## Source data

- aggregate JSON: `juyi-finetune:~/gen_out/round5_step100_n24_20260430T220301Z/round5_step100_agg.json`
- summary JSON (stats.py canonical): `juyi-finetune:/tmp/round5_step100_summary.json`
- per-prompt scores: `~/gen_out/round5_step100_n24_20260430T220301Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos: `~/gen_out/round5_step100_n24_20260430T220301Z/20260430T220304Z/heldout_regen/<pid>/trained/...`
- baseline reuse: `~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
