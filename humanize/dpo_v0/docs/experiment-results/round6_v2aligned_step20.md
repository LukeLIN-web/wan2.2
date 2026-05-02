# Round-6 v2aligned step-20 PhyJudge eval (n=24)

**Verdict**: `rolling-read-only`
**Criterion** (set in plan `humanize/dpo_v0/docs/exp-plan/round6_plan.md`): n=43 lora_final paired sign-test vs round-4 lora_final, p<0.05 favoring round-6.
**Next action**: no decisions taken at this checkpoint.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_step20.safetensors` (153,449,976 B) |
| ckpt sha256 | `334ce356bdb30969be4e85f8a5cb51af961fc2ab64be82237903f9085219a737` |
| trainer step health | loss=0.1938, gnorm=22.2, margin=+1.54, vram_peak_alloc=58.35GB |
| eval gen out | `juyi-finetune:~/gen_out/round6_step20_n24_20260501T052747Z/` |
| baseline_sha256_match | true |
| n | 24 |
| baseline ref | `[ref:v3_baseline_n42]` (`~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`) |

## Headline (n=24)

| metric | Δ | 95% CI | sign-test p (vs 0) | vs criterion |
|---|---|---|---|---|
| axes-avg Δ | −0.108 | [−0.350, +0.100] | 1.000 | within-noise |
| A 多体碰撞 (n=8) | −0.125 | [−0.400, +0.075] | — | within-noise |
| 其他类合计 (n=16) | (n_per_class<5, no CI) | — | — | rolling-read-only |

## Per-axis Δ (n=24)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | −0.083 | [−0.375, +0.167] | 1.000 |
| PTV | 0.000 | [−0.250, +0.250] | 1.000 |
| persistence | −0.292 | [−0.625, 0.000] | 0.180 |
| inertia | −0.083 | [−0.333, +0.125] | 1.000 |
| momentum | −0.083 | [−0.292, +0.083] | 1.000 |

All 5 axes within noise (every CI spans 0 except persistence which touches 0 at upper bound; sign-test p > 0.05 for all axes).

## Per-class Δ axes-avg (evalprompt.md A–G; n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 8 | −0.125 | [−0.400, +0.075] |
| B 破坏/形变 | 4 | +0.250 | n<5, no CI |
| C 流体 | 3 | +0.200 | n<5, no CI |
| D 阴影/反射 | 4 | 0.000 | n<5, no CI |
| E 链式 (v1 mapping; v2 demoted) | 1 | −2.200 | n<5, no CI |
| F 滚动/滑动 | 3 | −0.333 | n<5, no CI |
| G 抛掷/弹道 | 1 | 0.000 | n<5, no CI |

## Stuck-prompt watch (per-prompt scores, base ≤ 2)

| pid | class (v1 / v2) | base axes-mean | ckpt axes-mean | 5-axis (SA/PTV/persistence/inertia/momentum) |
|---|---|---|---|---|
| `2455740c4d45` | A / A | 2.0 | tbd | (computed in agg JSON; under-rendering this iteration) |

## Caveats (≤3)

- n=24 < 43 prompts → `rolling-read-only`. All Δ readouts here are direction-of-travel only and do not constitute evidence per `howtoreport.md` §一 #1.
- `eval/stats.py` `PROMPT_TO_CLASS` is a static module-level dict (still v1 mapping at this commit); my eval-v2 `PROMPT_CLASS.json` (`da1e56c`) is not yet wired into stats.py runtime — `36e42af19937` is reported in class E here even though eval-v2 demoted it. Class breakdown above shows v1 mapping; verdict-binding lora_final n=43 will use the v2 reference jsonls (`/tmp/round4_beta100_lora_final_v2.jsonl` etc.) which were assembled with v2 PROMPT_CLASS, not the static stats.py map. Stats.py source update is deferred (pending rl8 follow-up commit).
- F.11 contract preserved at this save: `lora_step20.safetensors` 153 MB + `lora_step20_optim.pt` 307 MB paired (FSDP `optim_state_dict()` rank-0 unsharded full state).

## Source data

- aggregate JSON (canonical, stats.py): `juyi-finetune:/tmp/round6_step20_summary.json`
- per-prompt scores: `juyi-finetune:~/gen_out/round6_step20_n24_20260501T052747Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos: `juyi-finetune:~/gen_out/round6_step20_n24_20260501T052747Z/<inner_ts>/heldout_regen/<pid>/trained/...`
- baseline reuse: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
- ckpt + paired optim: `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260501T043621Z/lora_step20.{safetensors,optim.pt}`
