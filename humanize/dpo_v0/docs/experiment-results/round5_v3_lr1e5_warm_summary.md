# Round-5 v3 lr1e5 warm — final summary (β=100 warm-start +1202 fresh pairs)

**Verdict**: `fail`
**Criterion** (locked pre-result `humanize/dpo_v0/docs/exp-plan/round5_eval_criterion.md`): n=42 lora_final paired sign-test vs round-4 lora_final, p<0.05 favoring round-5.
**Next action**: revert canonical winner to round-4 β=100 lora_final; eval-v2 transition in progress per `humanize/dpo_v0/docs/eval/CHANGELOG.md`.

## Run identity

| field | value |
|---|---|
| init LoRA | round-4 β=100 lora_final (`ckpts/20260429T234925Z/lora_final.safetensors`) |
| training data | 1202 cond_image_present pair_ids (= 2745 raw setminus round-4 1k filtered), sha `680a7eec8090d48b` |
| recipe_id | `6bef6e104cdd3442` (unchanged from round-4) |
| training_config sha | `88cc04e3b8dc9100` (lr=1e-5, β=100, lora_rank=16, num_samples=1202, seed_namespace `round5-warm-tier_b-1202-cond-present`) |
| trainer commit | `caeac71` (rl9 `--init-lora-from` + `--save-optimizer-state` + detect-key wrapper) |
| topology | juyi-videorl 8-rank FSDP, target_steps=151, save_every=50, ~135 s/step |
| total wall | 5h37m42s; 0 halt / 0 OOM / 0 traceback |
| routing | 151/151 high_noise (round-4 invariant ✓) |
| baseline_sha256_match | true |

## Trajectory ckpts

| ckpt | sha256[:16] | step health (loss/gnorm/margin) | mds |
|---|---|---|---|
| `lora_step50.safetensors` | `00e522c870ac9e50…` | 0.7515 / 35.7 / −0.114 | `round5_v3_lr1e5_warm_step50.md` |
| `lora_step100.safetensors` | `9280fa5aefa1e7eb…` | 0.5681 / 10.4 / +0.268 | `round5_v3_lr1e5_warm_step100.md` |
| `lora_step150.safetensors` | (paired with lora_final, +2 steps) | — (rolling skip per protocol) | n/a |
| `lora_final.safetensors` (step-151) | `4e8ca27979a9c710…` | 1.5631 / 16.3 / −1.33 (last single-pair) | `round5_v3_lr1e5_warm_lora_final.md` |

`lora_final_optim.pt` paired (307 MB) at every save point — F.11 contract preserved across 4 saves; round-N+1 warm-resume protocol is enabled.

## Headline (lora_final n=42, eval-v1) — VERDICT BINDING

| field | value |
|---|---|
| paired sign-test (round-5 vs round-4 lora_final) | n_pos=4 / n_neg=19 / n_zero=19 |
| p (two-sided binomial) | **0.0026** |
| reject H₀ at α=0.05 | true |
| direction | favoring round-4 |
| **verdict** | **`fail`** (revert to round-4 lora_final canonical winner) |

### Round-5 lora_final vs baseline (informational, not verdict-binding)

| metric | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| axes-avg Δ (n=42) | −0.038 | [−0.181, +0.090] | 1.000 |

All 5 per-axis Δ within noise (every CI spans 0).

## Rolling reads (n=24, rolling-read-only)

| ckpt | n | axes-avg Δ | 95% CI | sign-test p |
|---|---|---|---|---|
| step-50 | 24 | +0.025 | [−0.133, +0.175] | 0.804 |
| step-100 | 24 | −0.008 | [−0.167, +0.158] | 1.000 |

Both `rolling-read-only`; no decisions taken at non-final ckpts per `howtoreport.md` §一 #1.

## Per-class Δ axes-avg (lora_final n=42, eval-v1; n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 12 | +0.017 | [−0.150, +0.200] |
| B 破坏/形变 | 9 | −0.044 | [−0.400, +0.311] |
| C 流体 | 6 | +0.200 | [0.000, +0.400] |
| D 阴影/反射 | 5 | +0.080 | [0.000, +0.160] |
| E 链式 | 3 | −0.667 | n<5, no CI |
| F 滚动/滑动 | 4 | −0.200 | n<5, no CI |
| G 抛掷/弹道 | 2 | −0.100 | n<5, no CI |

Bonferroni-adjusted α for 5×7 = 0.00143; none of the per-class CIs exclude 0 at this corrected level. C / D class CI lower bounds touch 0 at uncorrected α=0.05 (informational). Per-class signals are **not verdict-binding**; the verdict is bound only to the all-n=42 paired sign-test in the Headline.

## Stuck-prompt watch (per-prompt scores, base ≤ 2)

| pid | class | base axes-mean | round-5 lora_final axes-mean | 5-axis (SA/PTV/persistence/inertia/momentum) |
|---|---|---|---|---|
| `2455740c4d45` | A | 2.0 | 1.4 | base 2/2/2/2/2 → ckpt 1/2/2/1/1 |

## Caveats (≤3)

- Eval-v1 prompt set used; eval-v2 transition is in progress and round-5 lora_final on the v2 set will be re-aggregated as a follow-up. The verdict above is binding for the round-5 ckpt under eval-v1. Per `howtoreport.md` §一 #2, cross-set comparisons across v1/v2 are forbidden — eval-v2 numbers (when published) are NOT directly comparable to the v1 numbers in this summary.
- AdamW init from scratch: round-4 lora_final saved only LoRA weights, so round-5 inits AdamW from zero. The first ~5–10 steps see momentum-buffer warmup; my `--save-optimizer-state` patch (caeac71) pairs an `_optim.pt` with each save going forward so round-N+1 can warm-resume momentum.
- `evalprompt.md` v1 G class header said n=3 but actually mapped 2/3 prompts in `PROMPT_CLASS.json` (one prompt was uncategorized in v1). Fixed in eval-v2 (`da1e56c`) — `b69b73d7b65f` re-included.

## Source data

- locked criterion: `humanize/dpo_v0/docs/exp-plan/round5_eval_criterion.md` (commit `7fc498b`)
- per-step mds: `round5_v3_lr1e5_warm_step{50,100}.md` + `round5_v3_lr1e5_warm_lora_final.md`
- aggregate JSON (canonical, stats.py): `juyi-finetune:/tmp/round5_lora_final_summary.json`
- paired sign-test JSON: `juyi-finetune:/tmp/paired_r5_vs_r4.json`
- round-5 trained scores (flat per-axis): `juyi-finetune:/tmp/round5_lora_final_trained_flat.jsonl`
- round-4 trained scores (flat per-axis, from #48 agg): `juyi-finetune:/tmp/round4_beta100_lora_final_trained_flat.jsonl`
- round-5 per-prompt scores: `juyi-finetune:~/gen_out/round5_lora_final_n42_20260501T000726Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos: `juyi-finetune:~/gen_out/round5_lora_final_n42_20260501T000726Z/20260501T000729Z/heldout_regen/<pid>/trained/...`
- baseline reuse: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
- eval-v2 transition: `humanize/dpo_v0/docs/eval/CHANGELOG.md`
