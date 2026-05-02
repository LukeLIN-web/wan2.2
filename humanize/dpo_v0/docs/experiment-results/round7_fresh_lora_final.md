# Round-7 fresh-init lora_final PhyJudge eval (n=40, eval-v2 first_alpha 42)

**Verdict**: `fail`
**Criterion** (locked pre-result `humanize/dpo_v0/docs/exp-plan/round7draft.md`): paired sign-test (round-7 lora_final, round-4 β=100 lora_final) on aligned v2 prompt-ids, p<0.05 favoring round-7.
**Next action**: canonical winner stays round-4 β=100 lora_final.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_final.safetensors` (153,449,976 B; step-128 = 100% of 1-epoch 1024-pair budget) |
| ckpt sha256 | `86bdc863cfd682e4c36666046510246561139ba13f59c1a2a8fa75d28932f2af` |
| trainer step health | step 128/128 high-noise routing 128/128 (100%) ✓, fraction_high_noise 1.0, total_forwards 512, 0 halt / 0 OOM / 0 traceback, 4h47m14s wall, vram_peak 58.35 GB alloc / 69.61 GB reserved |
| init LoRA | **fresh from base** (no `--init-lora-from`; A 路 design) |
| training data | 1024-pair v2-aligned subset (982 unique + 42 B-class repeats per decision-1 fallback ii); class quotas A:357/B:214/C:143/D:119/E:24/F:95/G:72=1024 |
| recipe | `recipes/training_config_round7_fresh_v2aligned_beta100.yaml` (sha `3431fd826fc5662a`); β=100, lr=1e-5, lora_rank=16, lora_alpha=16, num_samples=1024, target_steps=128 (1 epoch, zero wrap), save_every=32, sampling_band=[901,999], dpo_loss_kind=sigmoid |
| recipe_id | `6bef6e104cdd3442` (inherited from round-2 v0) |
| pair_ids_sha256_hex16 | `a262b7153f58c37f` |
| seed_namespace | `round7-fresh-tier_b-1024-cond-present` |
| eval gen out | `juyi-videorl:~/gen_out/round7_lora_final_n42_20260502T045255Z/` (regen) → tar-piped to juyi-finetune for score |
| eval score out | `juyi-finetune:~/gen_out/round7_lora_final_n42_20260502T045255Z/scores_perprompt/` |
| baseline_sha256_match | true (reuses `gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z`) |
| n | 40 (round-7 first_alpha 42 ∩ round-4 ref 43) |
| baseline ref | `/tmp/v3_baseline_v2.jsonl` (sha `02cd4a3899c8eb47`) |
| round-4 ref | `/tmp/round4_beta100_lora_final_v2.jsonl` (sha `e87a140dec375b69`) |

## Headline (n=40)

### Paired sign-test (round-7 lora_final vs round-4 β=100 lora_final, criterion-binding)

| field | value |
|---|---|
| n_aligned | 40 |
| n_pos (round-7 better) | 1 |
| n_neg (round-4 better) | 13 |
| n_zero (tied) | 26 |
| p (two-sided binomial) | 0.0018 |
| reject H₀ at α=0.05 | true |
| direction | **favoring round-4** |
| **verdict** | **`fail`** |

> Test rejects H₀ (p=0.0018) but direction is favoring round-4. Per locked
> criterion (`pass` requires direction=favoring round-7), this is `fail`.
> Round-4 β=100 lora_final remains the canonical winner. Round-5 (warm-start
> +1202 cond-present pairs), round-6 (warm-start +800 v2-aligned class-rebalanced),
> and round-7 (fresh-init +1024 v2-aligned) have all failed to beat round-4
> lora_final on the paired sign-test. The fresh-init hypothesis is rejected.

### Round-7 lora_final vs baseline (n=40, informational, not verdict-binding)

| metric | Δ | 95% CI |
|---|---|---|
| axes-avg Δ | +0.020 | [−0.090, +0.130] |

Within noise (CI spans 0).

## Per-axis Δ (n=40, vs round-4 β=100 lora_final)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | −0.150 | [−0.275, −0.050] | 0.0312 |
| PTV | −0.225 | [−0.375, −0.075] | 0.0117 |
| persistence | −0.050 | [−0.150, +0.050] | 0.6250 |
| inertia | +0.025 | [−0.150, +0.200] | 1.0000 |
| momentum | −0.075 | [−0.200, +0.050] | 0.4531 |

SA and PTV CIs exclude 0 favoring round-4 (uncorrected α=0.05). After Bonferroni for 5 axes: α=0.01, only PTV (p=0.0117) marginally significant. Per-axis-corrected pattern: round-4 is moderately better on SA + PTV; persistence/inertia/momentum within noise.

## Per-class Δ axes-avg (round-7 − round-4 β=100 lora_final, n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 12 | −0.083 | [−0.300, +0.100] |
| B 破坏/形变 | 9 | −0.156 | [−0.267, −0.067] |
| C 流体 | 6 | +0.033 | [−0.200, +0.233] |
| D 阴影/反射 | 5 | −0.080 | [−0.280, +0.080] |
| E 链式 | 1 | +0.000 | n<5, no CI |
| F 滚动/滑动 | 4 | −0.150 | n<5, no CI |
| G 抛掷/弹道 | 3 | −0.200 | n<5, no CI |

Bonferroni-adjusted α for 5×7=35 = 0.00143; none of the per-class CIs reject at this corrected level. B class CI excludes 0 at uncorrected α=0.05 favoring round-4.

## Step-trajectory paired sign-test vs round-4 (descriptive, not verdict-binding)

Round-7 trajectory descriptive points per `howtoreport.md` §一 #1; not verdict-binding. Each step regen used `selection_rule=first_alpha n_selections=42`, paired vs round-4 ref on 40 common prompts.

| ckpt | n | n_pos | n_neg | n_zero | p | direction | Δ axes-avg | CI |
|---|---|---|---|---|---|---|---|---|
| step32 | 40 | 7 | 18 | 15 | 0.0433 | favoring round-4 | −0.115 | [−0.225, −0.005] |
| step64 | 40 | 2 | 17 | 21 | 0.0007 | favoring round-4 | −0.190 | [−0.300, −0.075] |
| step96 | 40 | 6 | 20 | 14 | 0.0094 | favoring round-4 | −0.120 | [−0.235, −0.000] |
| **lora_final** | **40** | **1** | **13** | **26** | **0.0018** | **favoring round-4** | **−0.095** | **[−0.190, −0.010]** |

Trajectory monotone-ish negative throughout: step-64 peak loss (Δ −0.190), recovers slightly through step-96 → lora_final (Δ −0.095). All four ckpt points reject H₀ vs round-4. **Fresh-init from base does not converge to a winning trajectory at this 1024-pair v2-aligned budget.**

## Stuck-prompt watch (per-prompt scores, base axes-mean ≤ 2)

| pid | class | base axes-mean | ckpt axes-mean | base 5-axis | ckpt 5-axis |
|---|---|---|---|---|---|
| `1a0d4f1d8b1a` | A | 2.0 | 2.2 | 2/2/2/2/2 | 2/2/3/2/2 |
| `242e01f46c08` | B | 2.0 | 3.0 | 2/2/2/2/2 | 3/3/3/3/3 |
| `2455740c4d45` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `24d86e4e0339` | A | 2.0 | 1.0 | 2/2/2/2/2 | 1/1/1/1/1 |
| `80cc85fa7fa7` | F | 1.8 | 2.0 | 1/2/2/2/2 | 1/2/3/2/2 |
| `8f8b14d04c41` | B | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `ad664fa349ef` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `e0dae745a2a3` | A | 1.4 | 2.4 | 1/2/2/1/1 | 1/3/3/3/2 |
| `e90b3f54bffb` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |

Mixed: `24d86e4e0339` regressed (A 2→1), `242e01f46c08` (B 2→3) and `e0dae745a2a3` (A 1.4→2.4) improved; rest tied at 2.

## Caveats (≤4)

- **Selection mismatch**: round-7 step regens + lora_final regen used `selection_rule=first_alpha n_selections=42` from heldout_regen.py, which excludes 3 new eval-v2 A pids (`366d2a1252b3`, `5b7bb71f101d`, `d8b29a78eed7`) and includes 2 pids (`36e42af19937`, `9d500eec2188`) not in round-4 reference. Common-pair n=40 (not 43). The 3 new eval-v2 A pids were the round-6 verdict's distinctive contribution and aren't part of round-7's measurement set; this is a procedural drift from round-6 plan, not a deliberate choice. Round-8 should explicitly align to v2 n=43 for direct comparison.
- E / F / G class n<5 → raw Δ only, no CI. The class-rebalance plan brought E to n=1 in eval-v2 (per `docs/eval/CHANGELOG.md`), so E and G remain particularly low-power for any per-class signal.
- Round-7 lora_final regen on juyi-videorl 8-rank had a +1h13m idle-gap between regen done (~06:10 UTC) and score launch (~07:24 UTC) due to a wrong sentinel pattern in the rl9 progress watcher. Score itself ran cleanly (~21 min wall). Verdict numerics unaffected.
- `seed_namespace` is reported as `null` in the trainer's run_manifest.json (post-hoc field omission, not a recipe change); the recipe yaml still pins `seed_namespace: round7-fresh-tier_b-1024-cond-present`.

## Source data

- raw scores: `juyi-finetune:~/gen_out/round7_lora_final_n42_20260502T045255Z/scores_perprompt/<pid>/<TS>/summary.json` (42 prompts × 5 axes)
- aggregated jsonl: `juyi-finetune:/tmp/round7_lora_final_v2.jsonl` (210 rows = 42 × 5)
- step jsonls: `/tmp/round7_step{32,64,96}_v2.jsonl` (210 rows each)
- aggregator: `juyi-finetune:/tmp/round7_aggregate.py` (rl9, this run)
- paired-signtest: `juyi-finetune:/tmp/round7_paired_signtest.py` (rl9, this run)
- run manifest: `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260501T181040Z/run_manifest.json`
- regen run manifest: `juyi-finetune:~/gen_out/round7_lora_final_n42_20260502T045255Z/20260502T045302Z/run_manifest.{json,rank0..7.json}`
