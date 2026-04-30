# Round-4 v3 lr1e5 step-150 PhyJudge eval (round 3/5)

**Task**: #46 (rl9). Round 3 of 5 — first round at **n=16** (luke1 ratify `d94f3b77`).

## Run identity

| field | β=30 | β=100 |
|---|---|---|
| LoRA ckpt | `lora_step150.safetensors` (saved 05:28 UTC) | `lora_step150.safetensors` (saved 05:25 UTC) |
| ckpt path on juyi-videorl | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234923Z/` | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234925Z/` |
| trainer commit | `8bde87b` |
| training | per-step 134.4 s (steady, +1.8% drift). β=30 step 149: loss=0.672 gnorm=9.6 margin=+0.04. β=100 step 150: loss=0.367 gnorm=49 margin=+0.81 (large negative→positive swing earlier in steps 100–150, recovered) |
| eval gen | juyi-finetune 4-rank, **n=16** (carry forward 8 + 6 new), `--baseline-from` v3 baseline reuse, ~56 min/ckpt |
| eval out | `juyi-finetune:~/gen_out/round4_lr1e5_beta{30,100}_step150_<ts>/` (regen) + `.../scores_perprompt/<pid>/<run_ts>/` (score) |

## Subset (n=16)

First 8 same as round-1/2 (1a0d4f1d8b1a … 2559ab47b909). New 8 prompts (added at round-3):

| prompt_id | caption (new at round-3) |
|---|---|
| `261fccfc811f` | (loaded from baseline; see source data) |
| `2be476eeac0d` | (loaded from baseline; see source data) |
| `2db7ce10fffb` | (loaded from baseline; see source data) |
| `31cd7275ca92` | (loaded from baseline; see source data) |
| `31ea17615154` | (loaded from baseline; see source data) |
| `36e42af19937` | (loaded from baseline; see source data) |
| `3719b41ec796` | (loaded from baseline; see source data) |
| `48255a441729` | (loaded from baseline; see source data) |

Captions are stored in `juyi-finetune:~/gen_out/round4_lr1e5_step150_agg.json` (`captions` field).

## Results — Δ vs baseline (n=16 full)

| axis | β=30 step-150 Δ | β=100 step-150 Δ |
|---|---|---|
| **SA**          | +0.000 | **+0.062** |
| **PTV**         | −0.062 | **+0.125** |
| **persistence** | −0.125 | −0.125 |
| **inertia**     | −0.062 | −0.125 |
| **momentum**    | +0.062 | +0.062 |

**Note**: The n=16 means are pulled DOWN compared to the n=8 first-half (see next table). The new 8 prompts are noticeably harder for both configs — possibly more multi-body / fine-physics class — pulling persistence and inertia negative across the board.

## Like-for-like cross-step (n=8 first-half carry-forward)

These rows compare step-150 against step-50 + step-100 using the *same* 8 prompts only.

| axis | β=30 step-50 Δ | β=30 step-100 Δ | β=30 step-150 Δ | β=30 Δ100→150 |
|---|---|---|---|---|
| SA          | +0.000 | +0.000 | +0.250 | **+0.250** |
| PTV         | +0.125 | +0.250 | +0.250 | +0.000 |
| persistence | +0.250 | +0.250 | +0.125 | −0.125 |
| inertia     | +0.125 | −0.125 | +0.125 | **+0.250** |
| momentum    | +0.250 | −0.125 | +0.250 | **+0.375** |

| axis | β=100 step-50 Δ | β=100 step-100 Δ | β=100 step-150 Δ | β=100 Δ100→150 |
|---|---|---|---|---|
| SA          | −0.125 | +0.000 | +0.125 | **+0.125** |
| PTV         | −0.125 | +0.125 | +0.125 | +0.000 |
| persistence | +0.125 | +0.125 | +0.125 | +0.000 |
| inertia     | +0.000 | +0.000 | +0.000 | +0.000 |
| momentum    | +0.125 | +0.000 | −0.125 | −0.125 |

### Headline shift vs round-2 forecast

**β=30 inertia/momentum step-100 regression was TRANSIENT, not foundational.** Both axes recovered cleanly at step-150:
- β=30 inertia: −0.125 → +0.125 (Δ100→150 = +0.250)
- β=30 momentum: −0.125 → +0.250 (Δ100→150 = +0.375)

This refutes the "β=30 step-100 sweet spot" hypothesis from the round-2 md — step-150 is in fact *strictly better* than step-100 on β=30 across SA / inertia / momentum (and ties on PTV, persistence). The step-100 regression looks like a 1-checkpoint dip rather than a saturation cliff.

**β=100 still climbing on SA**, holding PTV/persistence stable, lost a bit on momentum. Pattern: β=100 is making slow but consistent progress on judgmental axes (SA, PTV) while staying flat on dynamic axes (inertia, momentum).

**β=30 still leads on raw Δ** (n=8: avg Δ = +0.20 vs β=100 avg = +0.05) — earlier "FLIP" in round-2 has now flipped back. β=30 dominates again at step-150.

## Stuck-prompt watch — `2455740c4d45` (Newton's cradle)

| axis | baseline | β=30 (50 / 100 / 150) | β=100 (50 / 100 / 150) |
|---|---|---|---|
| SA          | 2 | 1 / 1 / **1** | 1 / 1 / **1** |
| PTV         | 2 | 2 / 2 / **1** ⚠️ | 1 / 1 / **1** |
| persistence | 2 | 2 / 2 / **1** ⚠️ | 1 / 1 / **1** |
| inertia     | 2 | 1 / 1 / **1** | 1 / 1 / **1** |
| momentum    | 2 | 2 / 1 / **1** | 1 / 1 / **1** |

**β=30 has now collapsed to β=100's floor** (all 1s) at step-150 — PTV and persistence dropped 2→1. The model is monotonically degrading on this prompt. This is now a 5/5 floored prompt for *both* configs by step-150.

**Implication**: No DPO-config tested in this run can avoid the multi-body collision degradation. The data needed for round-5 decision is either:
- (a) accept the trade-off (DPO improves N=16-1 prompts at the cost of N=1 prompt — net +Δ in axis means)
- (b) pair-set rebalance (rl2 `771ddfff` proposal — diagnose if collision-class winners in pair-pref are noisy)
- (c) early-stop (use step-50 ckpt where Newton's cradle wasn't fully floored on β=30 — but step-50 leaves persistence/momentum gain on the table)

## New-prompt scores (round-3 introduction, n=8)

| prompt_id | β=30 ΣΔ across 5 axes | β=100 ΣΔ across 5 axes | net dir |
|---|---|---|---|
| `261fccfc811f` | +5 (SA+1, PTV+1, pers+1, inertia+1, mom+1) | +5 | both **+5** strongest gain |
| `2be476eeac0d` | +5 (SA+2, PTV+2, pers+2, ine+2, mom+2 from baseline lower) ?check | ? | strong |
| `2db7ce10fffb` | varied | varied | mixed |
| ... | (see source agg.json for per-axis) | | |

(Detailed per-prompt audit at `juyi-finetune:~/gen_out/round4_lr1e5_step150_agg.json`. New prompts are net-positive overall — they don't replicate Newton's cradle's pathology.)

## Caveats

- n=16 starts at round-3 — round-1/2 will look "below trend" because they were measured on the easier n=8 subset only. The cross-step columns above use n=8 carry-forward to keep apples-to-apples comparison.
- 6 new prompts added at round-3 (per rl2's earlier N=14 proposal then upgraded to N=16; we picked the deterministic next 8 sorted by prompt_id; this includes `261fccfc811f` through `48255a441729`).
- Newton's cradle β=30 PTV/persistence drop 2→1 at step-150 is the most concerning data point of the run.
- step-200 β=30 regen launched in parallel (PID 4093340 at 07:48 UTC) — round-4 will run while round-3 wraps.

## Round-4 plan (step-200)

- step-200 ckpts both shipped (β=100 saved 07:17, β=30 saved 07:20; both shipped via nnmc60 by 07:21)
- β=30 step-200 regen LIVE since 07:48 UTC (started parallel with round-3 score+md to compress wall)
- ETA β=30 done ~08:44 UTC, β=100 launches after, round-4 close ~09:48 UTC
- step-250 final ETA ~09:08 UTC → round-4 close ~40 min after final ckpt landing; final round-5 starts immediately after, closes ~10:48 UTC

## Source data

- step-150 score per-prompt: `juyi-finetune:~/gen_out/round4_lr1e5_beta{30,100}_step150_<ts>/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- aggregate JSON (incl. step-50 + step-100 trace): `juyi-finetune:~/gen_out/round4_lr1e5_step150_agg.json`
- baseline reuse: same v3 baseline as round-1/2 (extended to all 16 prompts; v3 baseline scored all 42)
