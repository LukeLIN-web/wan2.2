# Round-5 v3 lr1e5 warm step-50 PhyJudge eval (n=24)

**Task**: #50 (rl9 ratifier + Phase 4 eval owner). Round 1 of 3 — luke1 directive
`2101fa73` "n=8 太少了, 多一点" → n=24 rolling cadence (luke1 trust-rl9 dispatch
`05c0fd8e`).

This is the first eval round of the round-5 warm-start training run: β=100
(round-4 winner `+0.114` axes-avg) lora_final (step-250) used as `--init-lora-from`,
then trained another 1202 cond-image-present pairs (round-2 raw 2745 setminus
round-4 1k filtered) for 151 steps on juyi-videorl 8-rank FSDP.

## Run identity

| field | value |
|---|---|
| LoRA ckpt | `lora_step50.safetensors` (153.4 MB, FSDP-strip applied at save) |
| ckpt sha256 | `00e522c870ac9e509d66c69b4dd5980ab1f1cfcaf05486c02bc92c687648a2d6` |
| ckpt path on juyi-videorl | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260430T181330Z/` |
| `lora_step50_optim.pt` | 307 MB ✓ (F.11 — `_FSDP.optim_state_dict()` rank-0 unsharded) |
| trainer commit | `caeac71` (rl9 `--init-lora-from` + `--save-optimizer-state` + detect-key) |
| training | juyi-videorl 8-rank FSDP, num_samples=1202 (yaml-driven), target_steps=151, save_every=50, β=100, lr=1e-5, lora_rank=16, seed_namespace `round5-warm-tier_b-1202-cond-present`. Per-step ~135 s. Step-50 health: loss=0.7515, gnorm=35.7, margin=−0.114 (DPO not yet in positive zone — early warmup) |
| eval gen | juyi-finetune 4-rank, `--limit-prompts 24`, `--baseline-from` v3 baseline reuse, `multi_gpu_inference_seed_parallel`. Per-rank wrapper `/tmp/heldout_regen_rankdev_wrapper.sh` sets `--device cuda:${LOCAL_RANK}` (heldout_regen.py default `cuda` collides at rank 0 → 4-rank OOM). |
| eval scoring | physground env on juyi-finetune, 5 axes (SA, PTV, persistence, inertia, momentum), per-prompt `--run-dir` invocation with `--skip-existing` to avoid `manifest.timestamp_utc` gen_ts collision (round-4 BL). |
| eval out | `juyi-finetune:~/gen_out/round5_step50_n24_20260430T202042Z/` (regen + manifest) + `.../scores_perprompt/<pid>/<run_ts>/` (score) |
| aggregate | `juyi-finetune:~/gen_out/round5_step50_n24_20260430T202042Z/round5_step50_agg.json` |

## Subset (24 prompts, deterministic by `prompt_id` first-alpha)

| prompt_id | caption (truncated to 80 char) |
|---|---|
| `1a0d4f1d8b1a` | Two players reach for a volley, their racquets meet, creating a visible impact. |
| `1a44aba35343` | A person carries a heavy bucket while wading through chest-deep water. |
| `1b1c06c5ff1c` | A 30lb kettlebell is slowly lowered on top of a yellow ceramic coffee mug placed |
| `242e01f46c08` | A large pumpkin is placed on a small, flimsy stool; the stool breaks, and the pu |
| `2455740c4d45` | A Newton's cradle device on the table and one of the metal balls is held up by a |
| `24d86e4e0339` | A bowling ball rolls down a polished wooden lane, hitting the pins at the end. |
| `252b84def499` | The person gently places an egg yolk in a measuring cup into a pot of boiling wa |
| `2559ab47b909` | A player uses a backhand shot to send a racquetball into the corner, the ball re |
| `261fccfc811f` | A person walks confidently across a rocky desert. Under the warm sunlight, the p |
| `2be476eeac0d` | A person squeezes a tube of toothpaste, a continuous stream emerging onto a toot |
| `2db7ce10fffb` | A dart lands on the wire separating two numbers on the dartboard, the player sco |
| `31cd7275ca92` | A hammer is thrown and bounces once before stopping; the bounce and final restin |
| `31ea17615154` | A car's hood, propped open with a stick, is lifted from the other side; the stic |
| `36e42af19937` | Two rows of alternating black and white dominoes are set up on a wooden table wi |
| `3719b41ec796` | A close-up shot of a paddle making contact with the ping pong ball, showing the |
| `48255a441729` | A skater pushes off the wall of the ice rink, gaining momentum before beginning |
| `488e8d91cff5` | A white boat glides down the center of a narrow canyon surrounded by towering ro |
| `58db668bc142` | A hand presses a knife down through the center of a glossy, bright green gelatin |
| `5f68f5951b6b` | A metallic hurdle is struck by a hurdler's foot, creating a visible metallic cla |
| `5fdbe9f87762` | A player hits a grounder, the ball rolling smoothly across the perfectly manicur |
| `61345a00dfb5` | A sledgehammer is swung at a pile of rocks, scattering and breaking some of them |
| `6b48a3f28874` | A bulky, dark metallic robot figure stands enclosed behind a clear glass pane. A |
| `70d3b1b89e19` | A steel ball bearing hits a metal plate at a high speed, leaving a visible dent. |
| `75f6acbf5ba7` | A black car drives forward along a dusty desert road. The low sunlight casts a l |

## Results — score means (n=24, integer 1–4 PhyJudge scale)

| axis | v3 baseline | round-5 step-50 (β=100 warm) |
|---|---|---|
| **SA**          | 2.792 ± 0.658 | 2.917 ± 0.717 |
| **PTV**         | 3.042 ± 0.690 | 3.125 ± 0.850 |
| **persistence** | 3.500 ± 0.780 | 3.375 ± 0.824 |
| **inertia**     | 2.917 ± 0.504 | 2.875 ± 0.680 |
| **momentum**    | 2.875 ± 0.537 | 2.958 ± 0.690 |

## Δ vs baseline (trained − baseline, per-prompt then mean, n=24)

| axis | step-50 Δ |
|---|---|
| **SA**          | **+0.125** ± 0.612 |
| **PTV**         | **+0.083** ± 0.584 |
| **persistence** | −0.125 ± 0.537 |
| **inertia**     | −0.042 ± 0.464 |
| **momentum**    | **+0.083** ± 0.408 |
| **axes-avg**    | **+0.025** |

## Per-prompt-class breakdown

| class | n | axes-avg Δ |
|---|---|---|
| all | 24 | **+0.025** |
| non-collision (excl. Newton's cradle) | 23 | **+0.070** |
| collision (Newton's cradle `2455740c4d45`) | 1 | **−1.000** |

Even **excluding Newton's cradle**, axes-avg is only +0.070 — substantially below
round-4 lora_final's n=42 +0.127 non-collision read. The warm-start has clearly
perturbed the round-4 lora_final's gains; recovery is expected at step-100/lora_final.

## Newton's cradle (`2455740c4d45`) — collision-class regression

| axis | baseline | round-4 β=100 lora_final (n=1) | round-5 step-50 |
|---|---|---|---|
| SA          | 2 | **2** ⭐ | **1** ⬇ |
| PTV         | 2 | **2** ⭐ | **1** ⬇ |
| persistence | 2 | **2** ⭐ | **1** ⬇ |
| inertia     | 2 | 1 | 1 |
| momentum    | 2 | 1 | 1 |

**All 5 axes collapsed to 1** at step-50, regressing the round-4 lora_final's
3-of-5 recovery (SA / PTV / persistence had recovered to 2 in #48 full-42). This
is the most concerning signal: warm-start + new AdamW (no inherited round-4
optimizer momentum) appears to specifically punish the collision-class prompt at
the warmup-transient ckpt.

→ **Watch step-100 / step-150 / lora_final** to see if Newton's cradle recovers
back to round-4 lora_final's 3/5 state, or stays floored. If it stays floored
through lora_final, that would be a signal that warm-start dynamics differ
qualitatively from the round-4 from-scratch trajectory's lora_final convergence.

## Observations

1. **Mild axes-avg gain (+0.025) is consistent with early-warmup transient.** Per
   rl1's prediction (`fee5965d`): "round-4 trainer 只存了 LoRA weights，没存
   optimizer state ... 副作用: round-5 头几 step optimizer warmup 会有抖动 — 跟
   round-4 step-50 抖动同性质". Round-4 step-50 had axes-avg Δ −0.025 (β=100)
   from-scratch start; round-5 step-50 +0.025 from a warm-started LoRA — both
   small-magnitude, both within early-warmup regime. The contract holds.
2. **SA + PTV + momentum positive, persistence + inertia negative.** Persistence
   axis is most worrying (−0.125) — it's the round-4 winning axis at lora_final
   (round-4 #48 had +0.024). Persistence loss at step-50 suggests that warm-start
   is initially eroding the very physical-continuity signal that round-4 had
   secured.
3. **Subset selection hits 24 prompts including the Newton's cradle.** The
   collision-class prompt is now in the rolling subset (was n=8 in round-4 round-1
   eval), so the rolling read can track it directly. We'll see if it recovers at
   step-100/lora_final.
4. **Warm-start signature: divergent per-axis behavior.** Unlike from-scratch
   training where all axes drift together at step-50 (round-4 β=100 all axes
   slightly negative), warm-start at step-50 shows positive on 3 axes and negative
   on 2 — the optimizer is finding different per-axis trade-offs starting from
   the round-4 lora_final initialization.
5. **Sample size n=24 still has noise but is workable.** σ ≈ 0.4–0.6 per axis;
   means like ±0.125 are 0.2–0.3 σ events; meaningful as direction-of-travel
   across rolling cadence, not as standalone significance.

## Caveats

- **Subset n=24, NOT full 42.** Per round-5 protocol: rolling at n=24 (step-50,
  step-100), full **n=42 mandatory at lora_final** for the canonical winner-vs-
  baseline read. Round-1/2 readings are direction-of-travel only.
- **No baseline re-compute.** v3 baseline videos at
  `~/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z/heldout_regen/...` (reused via `--baseline-from`)
  and baseline scores at
  `~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl`
  (reused; only the 24 prompts in this subset are read). luke1 directive
  `082b9294` ratified.
- **PhyJudge integer 1–4 quantization.** Δ values like +0.125 mean "1 of 8
  prompts moved by +1, the rest unchanged" or similar low-rank changes. Not
  continuous improvement.
- **AdamW state warmup transient is real.** Round-5 trainer initialized AdamW
  from scratch (round-4 lora_final saved only LoRA weights, not optim state — my
  `--save-optimizer-state` patch landed in the round-5 trainer for round-N+1
  continuation). Effect: first ~5–10 steps of momentum buffers ramp from zero.
  Step-50 reading reflects this mid-warmup state, not steady-state DPO progress.

## Round-2 plan (step-100, in flight)

| ckpt | regen launched | OUT_DIR | ETA |
|---|---|---|---|
| `lora_step100.safetensors` (sha `9280fa5aefa1e7eb...`) | 22:03 UTC real (= chat 15:03) | `juyi-finetune:~/gen_out/round5_step100_n24_20260430T220301Z/` | regen ~22:30 UTC, score+agg+md ~23:00 UTC |

**Expected step-100 reading**: trainer health at step 100 already shows clear
DPO learning (loss 0.5681, gnorm 10.4, margin +0.268 — all signals positive vs
step-50). Eval Δ should pull closer to round-4 lora_final's +0.114 than the
+0.025 step-50 reading. Newton's cradle: open question — recovery to 3/5 at
step-100 would suggest warm-start retraces round-4's path; staying floored would
suggest qualitatively different trajectory.

## Round-3 plan (lora_final n=42, mandatory)

After step-150 / lora_final ckpts land (~16:55 chat). Skip step-150 separate
round (lora_final lands +2 min after step-150). lora_final eval at **n=42 full**
per protocol step 3, ETA close ~19:25 chat (~02:25 UTC May 1).

Final summary md will compare round-5 lora_final (n=42) vs round-4 lora_final
(n=42 #48 baseline) to determine whether the warm-start +1202-pair training
**improved or regressed** the round-4 winner.
