# Round-6 v2aligned lora_final PhyJudge eval (n=43, eval-v2)

**Verdict**: `fail`
**Criterion** (locked pre-result `humanize/dpo_v0/docs/exp-plan/round6_plan.md`): paired sign-test (round-6 lora_final, round-4 β=100 lora_final) on identical 43 v2 prompt-ids, p<0.05 favoring round-6.
**Next action**: canonical winner stays round-4 β=100 lora_final.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_final.safetensors` (153,449,976 B; step-100 = 100% of 1-epoch budget) |
| ckpt sha256 | `b8dd521025540dd763f121e8d823c70ffcf603db3e39ed42004d02c8708522ce` |
| trainer step health | step 100/100 high-noise routing 100/100 ✓, 0 halt / 0 OOM / 0 traceback, 3h45m23s wall, vram_peak 58.35 GB |
| init LoRA | round-4 β=100 lora_final (`ckpts/20260429T234925Z/lora_final.safetensors`) via `--init-lora-from` |
| training data | 800-pair eval-v2-aligned subset (A:279/B:167/C:112/D:93/E:19/F:74/G:56=800) from cond-present 2202 pool, sha `2749aeb2bb192148` |
| recipe | `recipes/training_config_round6_v2aligned_beta100.yaml` (sha `fa4dcf26a8f8e3e7`); β=100, lr=1e-5, lora_rank=16, num_samples=800, target_steps=100 (1 epoch, zero wrap), save_every=20, seed_namespace `round6-v2aligned-tier_b-800-cond-present` |
| eval gen out (v1 42 prompts) | `juyi-finetune:~/gen_out/round6_lora_final_n43_20260501T123824Z/` |
| eval gen out (3 new v2 A pids) | `juyi-finetune:~/gen_out/round6_lora_final_3newpids_20260501T154307Z/` |
| baseline_sha256_match | true |
| n | 43 (40 v1-shared + 3 new eval-v2 A pids per `docs/eval/CHANGELOG.md`) |
| baseline ref | `[ref:v3_baseline_v2_n43]` (`/tmp/v3_baseline_v2.jsonl` sha `02cd4a3899c8eb47`) |
| round-4 ref | `[ref:r4_lora_final_v2_n43]` (`/tmp/round4_beta100_lora_final_v2.jsonl` sha `e87a140dec375b69`) |

## Headline (n=43)

### Paired sign-test (round-6 lora_final vs round-4 β=100 lora_final, criterion-binding)

| field | value |
|---|---|
| n_aligned | 43 |
| n_pos (round-6 better) | 5 |
| n_neg (round-4 better) | 19 |
| n_zero (tied) | 19 |
| p (two-sided binomial) | 0.0066 |
| reject H₀ at α=0.05 | true |
| direction | **favoring round-4** |
| **verdict** | **`fail`** |

> Test rejects H₀ (p=0.0066) but direction is favoring round-4. Per locked
> criterion (`pass` requires direction=favoring round-6), this is `fail`.
> Round-4 β=100 lora_final remains the canonical winner. Round-5 (warm-start
> +1202 cond-present pairs, eval-v1 + eval-v2 both `fail`) and round-6
> (warm-start +800 v2-aligned class-rebalanced pairs, eval-v2 `fail`) have
> both failed to beat round-4 lora_final on the paired sign-test.

### Round-6 lora_final vs baseline (n=43, informational, not verdict-binding)

| metric | Δ | 95% CI | sign-test p (vs 0) | vs criterion |
|---|---|---|---|---|
| axes-avg Δ | −0.056 | [−0.167, +0.060] | — | within-noise |

## Per-axis Δ (n=43, vs baseline)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | −0.047 | [−0.209, +0.116] | 0.774 |
| PTV | +0.047 | [−0.116, +0.233] | 1.000 |
| persistence | −0.116 | [−0.279, +0.023] | 0.227 |
| inertia | −0.047 | [−0.186, +0.093] | 0.754 |
| momentum | −0.116 | [−0.279, +0.047] | 0.344 |

All 5 axes within noise (every CI spans 0; every sign-test p > 0.05).

## Per-class Δ axes-avg (evalprompt.md A–G v2; n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 15 | +0.040 | [−0.133, +0.200] |
| B 破坏/形变 | 9 | +0.022 | [−0.244, +0.311] |
| C 流体 | 6 | +0.067 | [−0.133, +0.300] |
| D 阴影/反射 | 5 | −0.080 | [−0.360, +0.160] |
| E 链式 | 1 | −0.400 | n<5, no CI |
| F 滚动/滑动 | 4 | −0.250 | n<5, no CI |
| G 抛掷/弹道 | 3 | −0.600 | n<5, no CI |

Bonferroni-adjusted α for 5×7=35 = 0.00143; none of the per-class CIs exclude 0
at this corrected level. Per-class results are descriptive supplement only,
not verdict-binding.

## Stuck-prompt watch (per-prompt scores, base ≤ 2)

| pid | class | base axes-mean | ckpt axes-mean | base 5-axis | ckpt 5-axis |
|---|---|---|---|---|---|
| `2455740c4d45` | A | 2.0 | 1.4 | 2/2/2/2/2 | 2/2/1/1/1 |
| `ad664fa349ef` | A | 2.0 | 1.4 | 2/2/2/2/2 | 2/2/1/1/1 |
| `e0dae745a2a3` | A | 1.4 | 1.4 | 1/2/2/1/1 | 1/2/2/1/1 |
| `1a0d4f1d8b1a` | A | 2.0 | 2.4 | 2/2/2/2/2 | 2/3/3/2/2 |
| `242e01f46c08` | B | 2.0 | 3.0 | 2/2/2/2/2 | 3/3/3/3/3 |
| `24d86e4e0339` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `80cc85fa7fa7` | F | 1.8 | 1.6 | 1/2/2/2/2 | 1/2/2/1/2 |
| `8f8b14d04c41` | B | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `e90b3f54bffb` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |

## Caveats (≤3)

- E / F / G class n<5 → raw Δ only, no CI. The class-rebalance plan brought E down to n=1 (per `docs/eval/CHANGELOG.md` v2 transition), so E and G are particularly low-power for any per-class signal.
- Step-20 / step-40 rolling reads were `rolling-read-only` per `howtoreport.md` §一 #1; step-60/step-80 evaluated separately on juyi-videorl 8-rank (task #51) — see `round6_v2aligned_step{20,40,60,80}.md` for trajectory descriptive points; none of those bind the verdict.
- 3 of the n=43 prompts (`366d2a1252b3`, `5b7bb71f101d`, `d8b29a78eed7`) are eval-v2 new A pids whose round-4 lora_final reference scores were generated 2026-04-30 (eval-v2 step 1); the other 40 reuse round-4 lora_final scores from #48 full-42 validation (`ad8fd28`).

## Source data

- locked criterion: `humanize/dpo_v0/docs/exp-plan/round6_plan.md`
- round-6 v2 paired sign-test JSON: `juyi-finetune:/tmp/paired_r6_vs_r4_v2.json`
- round-6 lora_final v2 flat jsonl: `juyi-finetune:/tmp/round6_lora_final_v2_43_flat.jsonl` sha `ef1f00cecb2d40cc`
- round-4 lora_final v2 flat jsonl: `juyi-finetune:/tmp/round4_beta100_lora_final_v2.jsonl` sha `e87a140dec375b69`
- baseline v2 flat jsonl: `juyi-finetune:/tmp/v3_baseline_v2.jsonl` sha `02cd4a3899c8eb47`
- per-prompt v1-shared (40 pids): `juyi-finetune:~/gen_out/round6_lora_final_n43_20260501T123824Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- per-prompt 3-new (3 A pids): `juyi-finetune:~/gen_out/round6_lora_final_3newpids_20260501T154307Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos (v1-42): `juyi-finetune:~/gen_out/round6_lora_final_n43_20260501T123824Z/<inner_ts>/heldout_regen/<pid>/trained/...`
- gen videos (3-new): `juyi-finetune:~/gen_out/round6_lora_final_3newpids_20260501T154307Z/<inner_ts>/heldout_regen/<pid>/trained/...`
- baseline reuse (v1 40): `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/...`
- baseline 3-new: `juyi-finetune:~/gen_out/eval_v2_baseline_plus_r4_n3_20260501T031745Z/<inner_ts>/heldout_regen/<pid>/baseline/...`
- ckpt + paired optim: `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260501T043621Z/lora_final.{safetensors,_optim.pt}`
