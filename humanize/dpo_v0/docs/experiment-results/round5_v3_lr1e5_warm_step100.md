# Round-5 v3 lr1e5 warm step-100 PhyJudge eval (n=24)

**Task**: #50 (rl9). Round 2 of 3 — n=24 rolling cadence per luke1 trust-rl9
dispatch (`05c0fd8e`).

## TL;DR

| metric | value |
|---|---|
| **all (n=24) axes-avg Δ** | **−0.008** (essentially flat) |
| non-collision (n=23) axes-avg Δ | +0.009 |
| collision (Newton's cradle n=1) axes-avg Δ | −0.400 (**3/5 axes recovered to base 2**) |

**Newton's cradle 3/5 recovery confirms warm-start retraced round-4 trajectory
on collision-class** ✓ — but **non-collision axes-avg +0.009 is far below
round-4 lora_final's +0.127 non-collision** at n=42, signaling warm-start +1202
training is *not* outperforming round-4 lora_final on the rest of the prompts at
this intermediate ckpt.

Hypothesis update (per rl2 `e232d822` protocol watch): warmup hypothesis was
half-right (collision-class recovered) but the protocol's full criterion
(axes-avg > +0.025) failed → real LoRA-init-perturbation effect on
non-collision. Step-150 / lora_final reading is now the deciding test.

## Run identity

| field | value |
|---|---|
| LoRA ckpt | `lora_step100.safetensors` (153.4 MB) |
| ckpt sha256 | `9280fa5aefa1e7eba439f1a45f347a8be5a88332fbb092b98e4d1bd2446a443f` |
| ckpt path on juyi-videorl | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260430T181330Z/lora_step100.safetensors` |
| `lora_step100_optim.pt` | 307 MB ✓ (F.11 contract preserved across multi-save) |
| trainer step-100 health | loss=0.5681, gnorm=10.4, margin=+0.268, vram_peak=58.35GB. Δ vs step-50: loss −24%, gnorm −71%, margin moved from −0.114 to +0.268 — DPO clearly converging. |
| eval gen | juyi-finetune 4-rank, `--limit-prompts 24`, `--baseline-from`. master-port=29702 to avoid clash with prior round. |
| eval out | `juyi-finetune:~/gen_out/round5_step100_n24_20260430T220301Z/` |
| aggregate | `~/gen_out/round5_step100_n24_20260430T220301Z/round5_step100_agg.json` |

## Results — score means (n=24, integer 1–4 PhyJudge)

| axis | v3 baseline | round-5 step-100 |
|---|---|---|
| **SA**          | 2.792 ± 0.658 | 2.833 ± 0.565 |
| **PTV**         | 3.042 ± 0.690 | 3.000 ± 0.722 |
| **persistence** | 3.500 ± 0.780 | 3.458 ± 0.721 |
| **inertia**     | 2.917 ± 0.504 | 2.917 ± 0.717 |
| **momentum**    | 2.875 ± 0.537 | 2.875 ± 0.797 |

## Δ vs baseline (n=24)

| axis | step-50 Δ | **step-100 Δ** | Δ→Δ direction (50→100) |
|---|---|---|---|
| **SA**          | +0.125 ± 0.612 | **+0.042 ± 0.464** | weaker (still positive) |
| **PTV**         | +0.083 ± 0.584 | **−0.042 ± 0.550** | flipped negative |
| **persistence** | −0.125 ± 0.537 | **−0.042 ± 0.550** | improved (toward 0) |
| **inertia**     | −0.042 ± 0.464 | **+0.000 ± 0.590** | recovered to flat |
| **momentum**    | +0.083 ± 0.408 | **+0.000 ± 0.659** | weaker (toward 0) |
| **axes-avg**    | **+0.025** | **−0.008** | net slightly worse |

## Per-prompt-class breakdown

| class | n | step-50 axes-avg | **step-100 axes-avg** |
|---|---|---|---|
| all | 24 | +0.025 | **−0.008** |
| non-collision (excl. Newton's cradle) | 23 | +0.070 | **+0.009** |
| collision (Newton's cradle `2455740c4d45`) | 1 | −1.000 | **−0.400** |

Non-collision axes-avg fell from +0.070 to +0.009 (essentially zero gain over
baseline at step-100). Round-4 lora_final's non-collision n=42 was +0.127, so
round-5 step-100 is **far below the round-4 winner reference** on
non-collision prompts.

## Newton's cradle (`2455740c4d45`) — collision-class trajectory

| axis | baseline | round-4 β=100 lora_final | **round-5 step-50** | **round-5 step-100** |
|---|---|---|---|---|
| SA          | 2 | **2** ⭐ | 1 ⬇ | **2** ⭐ recovered |
| PTV         | 2 | **2** ⭐ | 1 ⬇ | **2** ⭐ recovered |
| persistence | 2 | **2** ⭐ | 1 ⬇ | **2** ⭐ recovered |
| inertia     | 2 | 1 | 1 | 1 |
| momentum    | 2 | 1 | 1 | 1 |

**3 / 5 collision-class axes recovered at step-100, matching round-4 lora_final
exactly.** Same axes (SA / PTV / persistence) recovered, same axes (inertia /
momentum) stayed floored. Warm-start has retraced round-4's collision-class
trajectory.

This validates **half** of the warmup hypothesis (collision-class behavior
recovers as warmup transient resolves), but **invalidates the other half**
(non-collision axes are NOT recovering toward round-4 lora_final level — they're
hovering near baseline).

## Observations

1. **Non-collision regression vs round-4 lora_final is the key signal.**
   round-4 lora_final n=42 non-collision: +0.127. round-5 step-100 n=24
   non-collision: +0.009. Round-4's hard-won non-collision gains have been
   substantially eroded by the warm-start training. This is NOT a warmup
   transient artifact — at step-100, gnorm has dropped from 35.7 to 10.4, loss
   from 0.7515 to 0.5681, margin moved positive. Trainer is converging, just
   converging away from round-4 lora_final's optimum on non-collision.
2. **PTV flip from +0.083 to −0.042** is the single most concerning per-axis
   move. PTV was the only round-4 lora_final axis with statistical significance
   (sign-test p=0.039 vs β=30 step-150 at n=42). Now drifting negative.
3. **Newton's cradle full recovery at step-100** is encouraging but only
   represents 1 prompt out of 24. The training is finding the round-4 collision-
   class basin but losing ground elsewhere.
4. **rl2 protocol watch: hypothesis weakened.** rl2 `e232d822` set step-100
   criteria as "collision recovery ≥1.4 axes-mean AND axes-avg > +0.025".
   Newton's cradle axes-mean = 1.6 (passes ≥1.4 ✓), but axes-avg = −0.008 (fails
   > +0.025 ✗). 1-of-2 → real perturbation, NOT just transient.
5. **Trajectory hint**: persistence and inertia are recovering toward baseline
   (step-50 −0.125 → step-100 −0.042 for persistence; step-50 −0.042 → step-100
   +0.000 for inertia). SA + PTV + momentum are weakening. The warm-start
   training is *redistributing* axis-level performance, not uniformly improving.

## Caveats

- **n=24 not n=42**: rolling read, direction-of-travel only. Round-3 lora_final
  n=42 mandatory for the final winner-vs-baseline + winner-vs-round-4-lora_final
  comparison.
- **Round-4 lora_final n=42 reference (+0.114 axes-avg, +0.127 non-collision)**
  is the strict comparison target. If round-5 lora_final n=42 doesn't beat that,
  round-5 was a net regression over round-4 (discard or revert).
- **PTV negative flip at step-100** could be small-sample volatility (24 prompts
  × integer 1–4 scale, σ ≈ 0.55). Need lora_final n=42 to confirm if PTV
  regression is real.
- **Optimizer-warmup transient still partially in play**: AdamW second-moment
  accumulator needs ~10-20 steps to stabilize; we're at step 100 so steady-state.
  But persistence axis still mid-recovery suggests some second-order effects
  (LR schedule? data-distribution shift in the +1202 fresh pairs?) may take more
  steps to settle.

## Round-3 plan (lora_final n=42 mandatory)

After step-150 ckpt lands ~16:53 chat (ETA confirmed by rl8) → lora_final
expected ~16:55 chat (step-151). Skip step-150 separate eval (lora_final is +2
min after, redundant). Run lora_final at full **n=42** per protocol (covers all
42 heldout prompts including the 18 that aren't in the n=24 rolling subset).

**Decision criteria for round-5 lora_final** (per round-4 #48 framework):
1. **Round-5 axes-avg ≥ +0.114** at n=42 → round-5 wins (β=100 warm-start
   improved over round-4 lora_final)
2. **Round-5 axes-avg < +0.114** at n=42 → round-5 lost (revert to round-4
   lora_final as canonical winner)
3. **PTV sign-test p<0.05 vs round-4 lora_final** → statistical confirmation of
   either direction
4. **Newton's cradle 3/5 vs 5/5 recovered** at lora_final → marginal advantage

Wall: regen ~75-80 min for n=42 (42/4 = ceil 11 prompts/rank × ~13 min) on
juyi-finetune 4-rank → score ~10 min → md ~10 min → close ~19:25 chat
(02:25 UTC May 1).

## Source data

- aggregate JSON: `juyi-finetune:~/gen_out/round5_step100_n24_20260430T220301Z/round5_step100_agg.json`
- per-prompt scores: `~/gen_out/round5_step100_n24_20260430T220301Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- gen videos: `~/gen_out/round5_step100_n24_20260430T220301Z/20260430T220304Z/heldout_regen/<pid>/trained/...`
- baseline reuse: `~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
