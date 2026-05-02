# Round-4 v3 lr1e5 step-100 PhyJudge eval (round 2/5)

**Task**: #46 (rl9). Round 2 of 5 — luke1 directive `dd2d6da3` "之后每 50step 你都要 eval".

## Run identity

| field | β=30 | β=100 |
|---|---|---|
| LoRA ckpt | `lora_step100.safetensors` (β=30, 153.4 MB) | `lora_step100.safetensors` (β=100, 153.4 MB) |
| ckpt path on juyi-videorl | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234923Z/` | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234925Z/` |
| trainer commit | `8bde87b` (rl9 #42 fsdp-strip applied at save → ckpts ship-clean) |
| training | juyi-videorl 4-rank parallel (GPU 0-3 β=30, GPU 4-7 β=100), per-step 135.4 s (+1.8% drift vs 133s baseline per rl8 `31ed13c3`); β=100 step-99 spike loss=2.24 gnorm=151 → step-100 recovery loss=0.756 gnorm=16 (high-β saturation event, 1-step recovery, no halt) |
| eval gen | juyi-finetune 4-rank, **n=8 same subset as round-1** (deterministic by prompt_id first-alpha), `--baseline-from` v3 baseline reuse |
| eval out | `juyi-finetune:~/gen_out/round4_lr1e5_beta{30,100}_step100_<ts>/` (regen) + `.../scores_perprompt/<pid>/<run_ts>/` (score) |

## Results — score means (n=8, integer 1–4 PhyJudge scale)

| axis | v3 baseline | β=30 step-100 | β=100 step-100 |
|---|---|---|---|
| **SA**          | 2.375 ± 0.518 | 2.375 ± 0.744 | 2.375 ± 0.744 |
| **PTV**         | 2.625 ± 0.744 | 2.875 ± 0.641 | 2.750 ± 0.886 |
| **persistence** | 2.875 ± 0.991 | 3.125 ± 0.835 | 3.000 ± 1.069 |
| **inertia**     | 2.500 ± 0.535 | 2.375 ± 0.916 | 2.500 ± 0.756 |
| **momentum**    | 2.500 ± 0.535 | 2.375 ± 0.916 | 2.500 ± 0.926 |

## Δ vs baseline (trained − baseline, step-100, n=8)

| axis | β=30 step-100 Δ | β=100 step-100 Δ |
|---|---|---|
| **SA**          | +0.000 ± 0.535 | +0.000 ± 0.756 |
| **PTV**         | +0.250 ± 0.707 | +0.125 ± 0.835 |
| **persistence** | +0.250 ± 0.463 | +0.125 ± 0.641 |
| **inertia**     | −0.125 ± 0.835 | +0.000 ± 0.535 |
| **momentum**    | −0.125 ± 0.641 | +0.000 ± 0.756 |

## Cross-step trend: Δ(step-100) − Δ(step-50)  (step-on-step movement)

| axis | β=30 Δstep-50→100 | β=100 Δstep-50→100 | reading |
|---|---|---|---|
| **SA**          | +0.000 | **+0.125** | β=100 recovery from step-50 regression; β=30 flat |
| **PTV**         | **+0.125** | **+0.250** | both improving; β=100 catches up |
| **persistence** | +0.000 | +0.000 | both flat at +0.25 / +0.125 — stable region |
| **inertia**     | **−0.250** | +0.000 | β=30 *regression* — overshoot warning |
| **momentum**    | **−0.375** | −0.125 | β=30 *regression* worse than inertia — overshoot warning |

**Headline shift vs round-1**: β=100 is no longer dominated. β=100 *recovered* SA + PTV (it had been negative at step-50). β=30 started *losing* inertia + momentum — early sign of high-β-style saturation creeping into the lower-β config too. **Direction-of-travel flip**: round-1 said "β=30 dominates", round-2 says "β=100 catches up; β=30 starting to overshoot dynamic axes".

## Stuck-prompt watch — `2455740c4d45` (Newton's cradle)

Per rl2 `713096f1` rec — track this prompt's per-axis trajectory across rounds.

| axis | baseline | β=30 step-50 | β=30 step-100 | β=100 step-50 | β=100 step-100 |
|---|---|---|---|---|---|
| SA          | 2 | 1 | 1 | 1 | 1 |
| PTV         | 2 | 2 | 2 | 1 | 1 |
| persistence | 2 | 2 | 2 | 1 | 1 |
| inertia     | 2 | 1 | 1 | 1 | 1 |
| momentum    | 2 | 2 | **1** | 1 | 1 |

Reading:
- β=100 has been *floored* across all 5 axes since step-50 (every cell = 1, the worst score). Catastrophic for this prompt — model lost the multi-body collision constraint.
- β=30 mostly held its step-50 position at step-100, but **lost momentum** (2 → 1). Now β=30 is also approaching β=100's floor.
- If step-150 still shows β=30 momentum at 1 (or β=30 PTV/persistence drops to 1), this prompt is unrecoverable for this run; flag for round-5 cut.

## Per-prompt Δ at step-100 (audit trail)

prompt order: 1a0d4f1d8b1a, 1a44aba35343, 1b1c06c5ff1c, 242e01f46c08, 2455740c4d45, 24d86e4e0339, 252b84def499, 2559ab47b909

| axis | β=30 step-100 Δ | β=100 step-100 Δ |
|---|---|---|
| SA          | `[0, 0, 0, +1, −1, 0, 0, 0]` | `[0, 0, 0, +1, −1, 0, 0, 0]` |
| PTV         | `[+1, +1, 0, +1, 0, 0, 0, 0]` | `[+1, +1, 0, +1, −1, 0, 0, 0]` |
| persistence | `[+1, 0, 0, +1, 0, 0, 0, 0]` | `[+1, 0, 0, 0, −1, 0, 0, 0]` |
| inertia     | `[+1, 0, −1, +1, −1, −1, 0, 0]` | `[+1, 0, 0, 0, −1, 0, 0, 0]` |
| momentum    | `[+1, 0, 0, 0, −1, −1, 0, 0]` | `[+1, 0, 0, +1, −1, −1, 0, 0]` |

## Observations

1. **β=100 step-99 saturation event was transient.** The rl8/rl2-noted spike (loss 0.756→2.24→0.756 in 2 steps) does not propagate into PhyJudge. β=100 step-100 trained means are at-or-above baseline on all 5 axes — clean recovery.
2. **β=30 starting to overshoot dynamic axes (inertia/momentum).** This is the *first* round-1→round-2 directional regression we've seen for β=30. If it continues to step-150, lower lr or early-stop becomes the round-5 takeaway.
3. **PTV is the most uniformly positive axis** for both configs at step-100 (β=30 +0.250, β=100 +0.125). It is also the first axis where β=30 and β=100 both improved over step-50.
4. **Newton's cradle (`2455740c4d45`) is unrecoverable so far.** β=100 has been at the score-1 floor since step-50; β=30 just lost momentum at step-100. This is the strongest stuck-prompt candidate for round-5 subset cut.
5. **Inter-config noise**: per-prompt std at step-100 is 0.7–1.1 on a 1–4 scale → Δ ≈ 0.125 still inside noise. Trends are direction-only; the cross-step Δ ("Δstep-50→100") is the more useful signal.

## Caveats

- Same n=8 subset as round-1 — direct cross-step comparability ✓.
- Round-3/4/5 will jump to **n=16** per luke1 `d94f3b77` ratify; round-1/2 stay as n=8 pilot. Future md will note the denominator change.
- Score gen_ts collision footgun mitigated by per-prompt `--run-dir` invocation (same workaround as round-1).
- β=100 step-99 spike could still propagate at step-150+ (`U-shape` per rl2 forecast `187189fa`). Watch SA/PTV trajectory closely round-3.

## Round-3 plan (step-150)

- step-150 ckpts ETA ~05:26 UTC (rl8 reminder `79d7d27d`) — push +1.8% drift = ~05:32 UTC
- ship → regen β=30 then β=100 at **n=16 (4 prompts/rank × 14 min × 2 = 112 min wall)**
- score 32 videos + md ~05:39 → 07:25 UTC
- Round-4 (step-200) starts ~07:25 (~21 min after step-200 ckpt 07:04 lands; cumulative slip per rl2 `7488c58b`)

## Source data

- step-100 score per-prompt: `juyi-finetune:~/gen_out/round4_lr1e5_beta{30,100}_step100_<ts>/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- aggregate JSON: `juyi-finetune:~/gen_out/round4_lr1e5_step100_agg.json` (includes step-50 cross-step diff)
- baseline reuse: same as round-1
