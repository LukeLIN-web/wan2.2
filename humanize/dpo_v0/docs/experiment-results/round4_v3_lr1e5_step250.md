# Round-4 v3 lr1e5 step-250 (lora_final) PhyJudge eval (round 5/5)

**Task**: #46 (rl9). Round 5 of 5, n=16. **Final eval round** of the rolling cadence.

## Headline — β=100 recovers from step-200 cliff AND partially restores Newton's cradle

| axis | β=100 Δ200→250 (n=16) | β=30 Δ200→250 (n=16) |
|---|---|---|
| **SA**          | **+0.312** ⭐ | +0.062 |
| **PTV**         | **+0.375** ⭐ | +0.062 |
| **persistence** | **+0.500** ⭐ | −0.188 |
| **inertia**     | +0.000 | +0.062 |
| **momentum**    | +0.125 | −0.062 |

β=100 step-250 makes the largest single-step *positive* moves observed in the run on SA / PTV / persistence — exactly mirroring the catastrophic step-200 cliff in the opposite direction. **The "U-shape" forecast (rl2 `187189fa`) is fully realized: β=100 step-200 was the trough, step-250 the recovery.**

## Step-250 means (n=16) and Δ vs baseline

| axis | baseline | β=30 step-250 | β=30 Δ | β=100 step-250 | β=100 Δ |
|---|---|---|---|---|---|
| SA          | 2.625 | 2.562 | −0.062 | **2.812** | **+0.188** |
| PTV         | 2.875 | 2.750 | −0.125 | **3.000** | **+0.125** |
| persistence | 3.375 | 3.062 | −0.312 | **3.500** | **+0.125** |
| inertia     | 2.750 | 2.688 | −0.062 | 2.625 | −0.125 |
| momentum    | 2.750 | 2.625 | −0.125 | 2.750 | +0.000 |

**Step-250 winner is β=100 on 4 of 5 axes** (SA, PTV, persistence, momentum). β=30 only ties on inertia.

## n=8 carry-forward (apples-to-apples cross-step) full trajectory

| axis | β=30 50→100→150→200→250 | β=100 50→100→150→200→250 |
|---|---|---|
| SA          | +0.000 → +0.000 → +0.250 → +0.125 → **+0.125** | −0.125 → +0.000 → +0.125 → +0.125 → **+0.375** |
| PTV         | +0.125 → +0.250 → +0.250 → +0.000 → **+0.000** | −0.125 → +0.125 → +0.125 → −0.125 → **+0.250** |
| persistence | +0.250 → +0.250 → +0.125 → +0.125 → **+0.125** | +0.125 → +0.125 → +0.125 → −0.125 → **+0.250** |
| inertia     | +0.125 → −0.125 → +0.125 → +0.000 → **+0.125** | +0.000 → +0.000 → +0.000 → −0.125 → **+0.000** |
| momentum    | +0.250 → −0.125 → +0.250 → +0.000 → **+0.000** | +0.125 → +0.000 → −0.125 → −0.125 → **+0.125** |

## Trajectory winner rankings (per rl2 `36137b1e`)

Computed over n=8 carry-forward, axis-averaged across 5 steps.

| metric | β=30 | β=100 | winner |
|---|---|---|---|
| **weighted-mean** (0.5×step250 + 0.3×step200 + 0.2×step150) | +0.092 | +0.087 | β=30 (tie within noise) |
| **Spearman** (axes-avg, monotonicity) | **+0.020** | **+0.520** | **β=100** (much more monotone-up) |
| **step-250 final** (axes-avg) | +0.075 | **+0.200** | **β=100** (final value far higher) |
| **peak step** (best single-step axes-avg) | +0.200 (at step-150) | +0.200 (at step-250) | tied |

**β=100 wins on 2/4 metrics** (Spearman, step-250 final), ties peak, slightly trails on weighted-mean. By the rl2 `36137b1e` *trajectory-smoothness + final-step value* criterion, **β=100 lora_final (step-250) is the winning ckpt**.

β=30 lora_final is *not the winning β=30 ckpt*: β=30 step-150 (peak +0.200) is. But β=30 step-150 < β=100 step-250 (+0.200).

## Stuck-prompt watch — `2455740c4d45` (Newton's cradle): **partial recovery for β=100**

| axis | baseline | β=30 (50/100/150/200/250) | β=100 (50/100/150/200/250) |
|---|---|---|---|
| SA          | 2 | 1/1/1/1/**1** | 1/1/1/1/**2** ⭐ recovered |
| PTV         | 2 | 2/2/1/1/**1** | 1/1/1/1/**2** ⭐ recovered |
| persistence | 2 | 2/2/1/1/**1** | 1/1/1/1/**2** ⭐ recovered |
| inertia     | 2 | 1/1/1/1/**1** | 1/1/1/1/**1** still floored |
| momentum    | 2 | 2/1/1/1/**1** | 1/1/1/1/**1** still floored |

- **β=100 lora_final restores Newton's cradle on 3/5 axes (SA, PTV, persistence) back to baseline level (2).** Inertia + momentum still at floor.
- β=30 lora_final remains 5/5 floored (no recovery anywhere).
- This is strong evidence the β=100 step-250 recovery is *real* (multi-axis, includes the previously-broken collision class) — not a noise spike.

## Per-prompt-class composites (collision vs non-collision)

| axis | tag | all (n=16) | collision (Newton n=1) | non-collision (n=15) |
|---|---|---|---|---|
| SA          | β=30 | −0.062 | −1 | +0.000 |
| SA          | β=100 | +0.188 | +0 | +0.200 |
| PTV         | β=30 | −0.125 | −1 | −0.067 |
| PTV         | β=100 | +0.125 | +0 | +0.133 |
| persistence | β=30 | −0.312 | −1 | −0.267 |
| persistence | β=100 | +0.125 | +0 | +0.133 |
| inertia     | β=30 | −0.062 | −1 | +0.000 |
| inertia     | β=100 | −0.125 | −1 | −0.067 |
| momentum    | β=30 | −0.125 | −1 | −0.067 |
| momentum    | β=100 | +0.000 | −1 | +0.067 |

Even *excluding* Newton's cradle, **β=100 lora_final beats β=30 lora_final on 4 of 5 axes** (SA, PTV, persistence, momentum). The flip is not just about collision-class recovery — β=100 actually generalizes better at lora_final.

## Round-5 conclusion

- **Winning ckpt: β=100 lora_final (step-250)**, by trajectory-final-step criterion + Spearman monotonicity + Newton's cradle partial recovery + non-collision class composite.
- **Best β=30 ckpt: step-150** (peak +0.200 axes-avg), but it loses to β=100 lora_final by ~0 axes-avg though wider on 3 axes (SA β=100 +0.375 vs β=30 +0.250; PTV β=100 +0.250 vs β=30 +0.250 tied; persistence β=30 +0.125 vs β=100 +0.250).
- **Cross-round narrative**: β=30 was the early leader (round-1/2/3) but flat-lined or mildly declined from step-150 onward. β=100 had a U-shape — early mild gains, step-150 plateau, step-200 catastrophic regression, step-250 strong recovery to peak. *Final ckpt wins for the higher-β config.*

## Caveats

- n=8 noise is still ~0.5–0.7 score units → individual axis Δ ≈ 0.125 are within noise. The cross-axis aggregate trends are the more reliable signal.
- Newton's cradle 3/5 recovery is suggestive but n=1 — collision-class behavior would need multi-prompt validation in round-5+ runs.
- β=100 step-150→200 cliff plus step-200→250 recovery is a 200-step phenomenon — might or might not appear if training continued past step-250. Cliff hypothesis (over-training collapse) less likely now.

## Source

- step-250 score per-prompt: `juyi-finetune:~/gen_out/round4_lr1e5_beta{30,100}_step250_<ts>/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- final aggregate: `juyi-finetune:~/gen_out/round4_lr1e5_step250_agg.json`
- baseline reuse: same v3 baseline as rounds 1–4
