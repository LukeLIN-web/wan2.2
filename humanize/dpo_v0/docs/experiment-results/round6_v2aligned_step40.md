# Round-6 v2aligned step-40 PhyJudge eval (n=24)

**Verdict**: `rolling-read-only`
**Criterion** (set in plan `humanize/dpo_v0/docs/exp-plan/round6_plan.md`): n=43 lora_final paired sign-test vs round-4 lora_final, p<0.05 favoring round-6.
**Next action**: no decisions taken at this checkpoint.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_step40.safetensors` (153,449,976 B) |
| ckpt sha256 | `2f514ad823870fe51e011c0e22f2071d1ff62c1d5e3fc83044f7348f2569fb41` |
| trainer step health | loss=0.4919, gnorm=9.65, margin=+0.453, vram_peak_alloc=58.35GB |
| eval gen out | `juyi-videorl:~/gen_out/round6_step40_n24_20260501T141650Z/` (regen, 8-rank) → `juyi-finetune:~/gen_out/round6_step40_n24_20260501T141650Z/` (score, cuda:1 parallel) |
| baseline_sha256_match | true |
| n | 24 |
| baseline ref | `[ref:v3_baseline_n42]` (`~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`) |

## Headline (n=24)

| metric | Δ | 95% CI | sign-test p (vs 0) | vs criterion |
|---|---|---|---|---|
| axes-avg Δ | −0.050 | [−0.242, +0.133] | 1.000 | within-noise |
| A 多体碰撞 (n=8) | −0.025 | [−0.350, +0.250] | — | within-noise |
| 其他类合计 (n=15) | (n_per_class<5, no CI) | — | — | rolling-read-only |

## Per-axis Δ (n=24)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | −0.083 | [−0.292, +0.125] | 0.688 |
| PTV | −0.042 | [−0.292, +0.208] | 1.000 |
| persistence | −0.125 | [−0.375, +0.125] | 0.727 |
| inertia | +0.042 | [−0.167, +0.250] | 1.000 |
| momentum | −0.042 | [−0.250, +0.167] | 1.000 |

All 5 axes within noise (every CI spans 0; every sign-test p > 0.05).

## Per-class Δ axes-avg (evalprompt.md A–G; n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 8 | −0.025 | [−0.350, +0.250] |
| B 破坏/形变 | 4 | +0.150 | n<5, no CI |
| C 流体 | 3 | +0.267 | n<5, no CI |
| D 阴影/反射 | 4 | +0.000 | n<5, no CI |
| E 链式 | 0 | n/a | n/a |
| F 滚动/滑动 | 3 | −0.267 | n<5, no CI |
| G 抛掷/弹道 | 1 | −0.400 | n<5, no CI |

## Stuck-prompt watch (per-prompt scores, base ≤ 2)

| pid | class | base axes-mean | ckpt axes-mean | 5-axis (SA/PTV/persistence/inertia/momentum) |
|---|---|---|---|---|
| `1a0d4f1d8b1a` | A | 2.0 | 2.6 | base 2/2/2/2/2 → ckpt 2/2/3/3/3 |
| `242e01f46c08` | B | 2.0 | 3.0 | base 2/2/2/2/2 → ckpt 3/3/3/3/3 |
| `2455740c4d45` | A | 2.0 | 1.0 | base 2/2/2/2/2 → ckpt 1/1/1/1/1 |
| `24d86e4e0339` | A | 2.0 | 2.0 | base 2/2/2/2/2 → ckpt 2/2/2/2/2 |

## Caveats (≤3)

- n=24 < 43 prompts → `rolling-read-only`. All Δ readouts here are direction-of-travel only and do not constitute evidence per `howtoreport.md` §一 #1.
- Per-class breakdown uses v2 `PROMPT_CLASS.json` (rl8 `fba8867`); class E shows n=0 because `36e42af19937` was demoted (NOT_PRESENT in v2). Step-20 md (`774b043`) caveated v1 mapping; this md is v2-aligned.
- F.11 contract preserved at this save: `lora_step40.safetensors` (153 MB) + `lora_step40_optim.pt` (307 MB) paired (FSDP `optim_state_dict()` rank-0 unsharded full state).

## Source data

- aggregate JSON (canonical, stats.py): `juyi-finetune:~/gen_out/round6_step40_n24_20260501T141650Z/round6_step40_agg.json`
- per-prompt scores: `juyi-finetune:~/gen_out/round6_step40_n24_20260501T141650Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos: `juyi-finetune:~/gen_out/round6_step40_n24_20260501T141650Z/20260501T141653Z/heldout_regen/<pid>/trained/...` (scp'd from `juyi-videorl:~/gen_out/round6_step40_n24_20260501T141650Z/` via tar-pipe)
- baseline reuse: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
- ckpt + paired optim: `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260501T043621Z/lora_step40.{safetensors,optim.pt}`
