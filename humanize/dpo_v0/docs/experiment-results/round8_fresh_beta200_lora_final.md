# Round-8 fresh-init β=200 lora_final PhyJudge eval (n=43, eval-v2)

**Verdict**: `fail`
**Criterion** (locked: clone of round-7 criterion with β=100→β=200): paired sign-test (round-8 lora_final, round-4 β=100 lora_final) on aligned v2 prompt-ids n=43, p<0.05 favoring round-8.
**Next action**: canonical winner stays round-4 β=100 lora_final.

## Run identity

| field | value |
|---|---|
| ckpt | `lora_final.safetensors` (153,449,976 B; step-128 = 100% of 1-epoch 1024-pair budget) |
| ckpt sha256 | `96ca273d7cbe856abcc7f4951857fa79cf761c3acbe34952d83fbf5d8754eeed` |
| trainer step health | step 128/128 high-noise routing 128/128 (100%) ✓, fraction_high_noise 1.0, total_forwards 512, 0 halt / 0 OOM / 0 traceback, 4h47m08s wall, vram_peak 58.35 GB alloc |
| init LoRA | **fresh from base** (no `--init-lora-from`; mirrors round-7 A 路) |
| training data | 1024-pair v2-aligned subset (982 unique + 42 B-class repeats per round-7 decision-1 fallback ii); class quotas A:357/B:214/C:143/D:119/E:24/F:95/G:72=1024 — **identical to round-7** |
| recipe | `recipes/training_config_round8_fresh_v2aligned_beta200.yaml` (sha `3d3ebb6907c48501`); β=**200** (delta vs round-7), lr=1e-5, lora_rank=16, lora_alpha=16, num_samples=1024, target_steps=128 (1 epoch), save_every=32 |
| recipe_id | `6bef6e104cdd3442` (carry-forward) |
| pair_ids_sha256_hex16 | `a262b7153f58c37f` (same as round-7) |
| seed_namespace | `round7-fresh-tier_b-1024-cond-present` (kept identical to round-7 for clean β-only ablation) |
| eval gen out (main 42) | `juyi-videorl:~/gen_out/round8_lora_final_n42_20260502T124042Z/` (regen) → tar-piped to juyi-finetune for score |
| eval gen out (3 new pids) | `juyi-finetune:~/gen_out/round8_lora_final_3newpids_20260502T144607Z/` (regen + score) |
| baseline_sha256_match | true for main 42 (reuses `gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z`); 3-pid baselines regenerated fresh (no v3 reuse for those 3 prompt-ids) |
| n | 43 (40 first_alpha v1-shared + 3 new eval-v2 A pids per `docs/eval/CHANGELOG.md` v2 = round-6 alignment) |
| baseline ref | `/tmp/v3_baseline_v2.jsonl` (sha `02cd4a3899c8eb47`) |
| round-4 ref | `/tmp/round4_beta100_lora_final_v2.jsonl` (sha `e87a140dec375b69`) |
| round-7 ref (for β-only ablation) | `/tmp/round7_lora_final_v2.jsonl` (n=42 first_alpha; rl9 commit `0ace236`) |

## Headline (n=43)

### Paired sign-test (round-8 lora_final β=200 vs round-4 β=100 lora_final, criterion-binding)

| field | value |
|---|---|
| n_aligned | 43 |
| n_pos (round-8 better) | 4 |
| n_neg (round-4 better) | 19 |
| n_zero (tied) | 20 |
| p (two-sided binomial) | 0.0026 |
| reject H₀ at α=0.05 | true |
| direction | **favoring round-4** |
| **verdict** | **`fail`** |

> Test rejects H₀ (p=0.0026) but direction is favoring round-4. Per locked
> criterion (`pass` requires direction=favoring round-8), this is `fail`.
> Round-4 β=100 lora_final remains the canonical winner. Round-5 (warm + 1202
> setminus, β=100), round-6 (warm + 800 v2-aligned, β=100), round-7 (fresh +
> 1024 v2-aligned, β=100), and round-8 (fresh + 1024 v2-aligned, **β=200**)
> have all failed to beat round-4 lora_final. The β=200 single-knob delta
> from round-7 does not flip the verdict.

### Round-8 lora_final β=200 vs baseline (n=43, informational)

| metric | Δ | 95% CI |
|---|---|---|
| axes-avg Δ | −0.003 | [−0.110, +0.105] |

Within noise (CI spans 0). Round-8 is approximately at-baseline overall.

### Round-8 vs round-7 (β-only ablation, n=42 first_alpha intersection)

| field | value |
|---|---|
| n_aligned | 42 |
| n_pos (round-8 better) | 12 |
| n_neg (round-7 better) | 8 |
| n_zero (tied) | 22 |
| p (two-sided binomial) | 0.5034 |
| reject H₀ at α=0.05 | false |
| direction | (insignificant) favoring round-8 β=200 |
| Δ_axes-avg | +0.010, CI=[−0.129, +0.157] |

> **β=200 vs β=100 is a TIE.** Round-8 numerically nudged 12-8-22 in its favor,
> but the test is far from significant (p=0.50). The β-only ablation answer is:
> **β=200 does not significantly improve over β=100 in this 1-epoch fresh-init
> regime.** This is consistent with round-8 step-32 already showing TIE
> (p=1.0, Δ−0.029) on the same paired set. Across the full 4-ckpt trajectory
> (step 32/64/96/lora_final), β=200 tracks β=100 within noise.

## Per-axis Δ (n=43, vs round-4 β=100 lora_final)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | −0.140 | [−0.256, −0.023] | 0.0703 |
| PTV | −0.163 | [−0.302, −0.047] | 0.0391 |
| persistence | −0.140 | [−0.279, +0.000] | 0.1460 |
| inertia | −0.116 | [−0.279, +0.047] | 0.3438 |
| momentum | −0.070 | [−0.233, +0.093] | 0.5488 |

SA and PTV CIs exclude 0 favoring round-4 (uncorrected α=0.05; only PTV reaches the per-axis sign-test α=0.05). After Bonferroni for 5 axes: α=0.01, none individually significant; the verdict signal is in the aggregate composite + axes-avg.

## Per-class Δ axes-avg (round-8 − round-4 β=100 lora_final, n<5 = raw Δ only)

| class | n | Δ | 95% CI |
|---|---|---|---|
| **A 多体碰撞** | **15** | **−0.147** | **[−0.307, −0.027]** |
| B 破坏/形变 | 9 | −0.133 | [−0.267, +0.000] |
| C 流体 | 6 | +0.133 | [−0.167, +0.433] |
| D 阴影/反射 | 5 | −0.000 | [−0.160, +0.200] |
| E 链式 | 1 | +0.200 | n<5, no CI |
| F 滚动/滑动 | 4 | −0.400 | n<5, no CI |
| G 抛掷/弹道 | 3 | −0.467 | n<5, no CI |

A class (15 prompts, the largest bucket) loses to round-4 with CI excluding 0. F + G are very negative but n<5. C class is the only positive, but CI spans 0 wide.

## Step-trajectory paired sign-test vs round-4 (descriptive, not verdict-binding)

Round-8 trajectory descriptive points; not verdict-binding. step32 paired vs round-4 ref on 40 common prompts (first_alpha 42 ∩ round-4 ref 43 = 40). Lora_final paired on n=43 (with C1 fix 3-pid supplement).

| ckpt | n | Δ_axes-avg | p vs round-4 | direction |
|---|---|---|---|---|
| step32 | 40 | −0.135 | 0.0266 | favoring round-4 |
| **lora_final** (step 128) | **43** | **−0.126** | **0.0026** | **favoring round-4** |

Step 64/96 ckpts saved on disk but rolling regen+score was deferred per cycle-time pressure (rl2 `73006fc0` consultation). β-only ablation pattern at step32 (round-8 vs round-7 TIE p=1.0) and at lora_final (TIE p=0.50) is internally consistent — β=200 does not change round-7's trajectory shape.

## Stuck-prompt watch (per-prompt scores, base axes-mean ≤ 2)

| pid | class | base axes-mean | ckpt axes-mean | base 5-axis | ckpt 5-axis |
|---|---|---|---|---|---|
| `1a0d4f1d8b1a` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `242e01f46c08` | B | 2.0 | **3.2** | 2/2/2/2/2 | 3/3/4/3/3 |
| `2455740c4d45` | A | 2.0 | **1.6** | 2/2/2/2/2 | 2/2/2/1/1 |
| `24d86e4e0339` | A | 2.0 | **1.8** | 2/2/2/2/2 | 2/2/2/2/1 |
| `80cc85fa7fa7` | F | 1.8 | **1.4** | 1/2/2/2/2 | 1/2/2/1/1 |
| `8f8b14d04c41` | B | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `ad664fa349ef` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |
| `e0dae745a2a3` | A | 1.4 | **1.0** | 1/2/2/1/1 | 1/1/1/1/1 |
| `e90b3f54bffb` | A | 2.0 | 2.0 | 2/2/2/2/2 | 2/2/2/2/2 |

`242e01f46c08` (B) the only clear improvement (2.0→3.2, recovers all 5 axes); `2455740c4d45` / `24d86e4e0339` / `80cc85fa7fa7` / `e0dae745a2a3` regressed (mostly inertia + momentum).

## Caveats (≤4)

- **n=43 alignment** restored vs round-7 verdict (which had n=40 due to first_alpha 42 missing 3 new eval-v2 A pids). Round-8 mirrors round-6 pattern: main first_alpha 42 regen + supplementary 3-pid regen on `/tmp/eval_v2_t0_t3_root` → aggregated to n=43 = full eval-v2 set. Direct comparable to round-6 verdict.
- 3-pid supplementary regen took 3 retries on juyi-finetune 4-rank: (1) `heldout pair count mismatch` → fixed with `--skip-heldout-count-assert`; (2) `--baseline-from missing` for new pids → dropped `--baseline-from` and regenerated fresh baselines; (3) success. 3-pid baselines therefore are NOT byte-identical to round-6 3-pid baselines (round-6 also did fresh baseline regen, but at different ts). Trained-side scoring only (paired vs round-4 jsonl ref) is unaffected.
- Step 64 / 96 rolling reads NOT executed for round-8 (deferred per scope-reduction at `fac78f33`/`73006fc0`). Trajectory has only step32 + lora_final descriptive points. β-only ablation is robust at the 2 measured ckpts; intermediate trajectory could be reconstructed if luke wants by running step64+96 regen+score on juyi-videorl 8-rank post-this-ship (~60 min).
- `seed_namespace` reported `null` in the trainer's run_manifest.json (post-hoc field omission, same C4 caveat as round-7 verdict md). Recipe yaml pins `seed_namespace: round7-fresh-tier_b-1024-cond-present` correctly; manifest writer needs a fix.

## Source data

- aggregated jsonl: `juyi-finetune:/tmp/round8_lora_final_v2.jsonl` (225 rows = 45 prompts × 5 axes; common-with-round-4 = 43)
- main 42 jsonl: `/tmp/round8_lora_final_n42.jsonl` (210 rows)
- 3-pid jsonl: `/tmp/round8_lora_final_3pid.jsonl` (15 rows)
- aggregator: `juyi-finetune:/tmp/round7_aggregate.py` (rl9, reused from round-7)
- paired-signtest: `juyi-finetune:/tmp/round7_paired_signtest.py` (rl9, reused from round-7)
- run manifest: `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260502T075054Z/run_manifest.json` (final_loss=0.5779, loss_mean=0.6789, vram_peak 58.35 GB, wall 17228 sec)
- regen run manifests: `juyi-finetune:~/gen_out/round8_lora_final_n42_*/run_manifest.rank{0..7}.json` + `round8_lora_final_3newpids_*/`
- analysis cross-link: `videodpo:docs/experiment-results/round7_vs_round4_analysis.md` (rl9 task #54, why round-7/8 lose to round-4: undertraining hypothesis = 1 epoch vs round-4's 2 epochs)
