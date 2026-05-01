# Round-5 v3 lr1e5 warm step-50 PhyJudge eval (n=24)

**Verdict**: `rolling-read-only`
**Criterion** (set in plan `humanize/dpo_v0/docs/exp-plan/round5_eval_criterion.md`): n=42 lora_final paired sign-test vs round-4 lora_final, p<0.05 favoring round-5.
**Next action**: no decisions taken at this checkpoint.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_step50.safetensors` (153,449,976 B) |
| ckpt sha256 | `00e522c870ac9e509d66c69b4dd5980ab1f1cfcaf05486c02bc92c687648a2d6` |
| trainer step health | loss=0.7515, gnorm=35.7, margin=−0.114, vram_peak_alloc=58.35GB |
| eval gen out | `juyi-finetune:~/gen_out/round5_step50_n24_20260430T202042Z/` |
| baseline_sha256_match | true |
| n | 24 |
| baseline ref | `[ref:v3_baseline_n42]` (`~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`) |

## Headline (n=24)

| metric | Δ | 95% CI | sign-test p (vs 0) | vs criterion |
|---|---|---|---|---|
| axes-avg Δ | +0.025 | [−0.133, +0.175] | 0.804 | within-noise |
| A 多体碰撞 (n=8) | −0.125 | [−0.400, +0.050] | — | within-noise |
| 其他类合计 (n=16) | (n_per_class<5, no CI) | — | — | rolling-read-only |

## Per-axis Δ (n=24)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | +0.125 | [−0.125, +0.375] | 0.508 |
| PTV | +0.083 | [−0.125, +0.333] | 0.727 |
| persistence | −0.125 | [−0.333, +0.083] | 0.453 |
| inertia | −0.042 | [−0.208, +0.125] | 1.000 |
| momentum | +0.083 | [−0.083, +0.250] | 0.625 |

All 5 axes within noise (every CI spans 0; every sign-test p > 0.05).

## Per-class Δ axes-avg (evalprompt.md A–G; n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 8 | −0.125 | [−0.400, +0.050] |
| B 破坏/形变 | 4 | +0.400 | n<5, no CI |
| C 流体 | 3 | +0.267 | n<5, no CI |
| D 阴影/反射 | 4 | +0.100 | n<5, no CI |
| E 链式 | 1 | −0.600 | n<5, no CI |
| F 滚动/滑动 | 3 | −0.200 | n<5, no CI |
| G 抛掷/弹道 | 1 | 0.000 | n<5, no CI |

## Stuck-prompt watch (per-prompt scores, base ≤ 2)

| pid | class | base axes-mean | ckpt axes-mean | 5-axis (SA/PTV/persistence/inertia/momentum) |
|---|---|---|---|---|
| `2455740c4d45` | A | 2.0 | 1.0 | base 2/2/2/2/2 → ckpt 1/1/1/1/1 |

## Caveats (≤3)

- n=24 < 42 prompts → `rolling-read-only`. All Δ readouts here are direction-of-travel only and do not constitute evidence per `howtoreport.md` §一 #1.
- Only A class has n≥5 in this rolling subset; B/C/D/E/F/G show raw Δ only (per `howtoreport.md` §二, n<5 ⇒ no CI).
- Round-4 lora_final reference [ref:r4_lora_final_n42] axes-avg lives in a different prompt set (n=42 vs n=24 here); cross-n point-estimate comparison is forbidden per `howtoreport.md` §一 #2 and is intentionally not shown.

## Source data

- aggregate JSON: `juyi-finetune:~/gen_out/round5_step50_n24_20260430T202042Z/round5_step50_agg.json`
- summary JSON (stats.py canonical): `juyi-finetune:/tmp/round5_step50_summary.json`
- per-prompt scores: `~/gen_out/round5_step50_n24_20260430T202042Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos: `~/gen_out/round5_step50_n24_20260430T202042Z/20260430T202045Z/heldout_regen/<pid>/trained/...`
- baseline reuse: `~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
