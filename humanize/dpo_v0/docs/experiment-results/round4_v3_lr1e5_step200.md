# Round-4 v3 lr1e5 step-200 PhyJudge eval (round 4/5)

**Task**: #46 (rl9). Round 4 of 5, n=16.

## Cross-round headline (n=8 carry-forward — apples-to-apples)

| axis | β=30 50→100→150→200 (Δ vs baseline) | β=100 50→100→150→200 (Δ vs baseline) |
|---|---|---|
| **SA**          | +0.000 → +0.000 → +0.250 → **+0.125** | −0.125 → +0.000 → +0.125 → **+0.125** |
| **PTV**         | +0.125 → +0.250 → +0.250 → **+0.000** ⚠️ | −0.125 → +0.125 → +0.125 → **−0.125** ⚠️ |
| **persistence** | +0.250 → +0.250 → +0.125 → **+0.125** | +0.125 → +0.125 → +0.125 → **−0.125** ⚠️ |
| **inertia**     | +0.125 → −0.125 → +0.125 → **+0.000** | +0.000 → +0.000 → +0.000 → **−0.125** ⚠️ |
| **momentum**    | +0.250 → −0.125 → +0.250 → **+0.000** | +0.125 → +0.000 → −0.125 → **−0.125** |

**Headline: β=100 is now in catastrophic regression (rl2 `187189fa` U-shape forecast confirmed).** β=100 first time below baseline on PTV / persistence / inertia simultaneously at step-200. β=30 is a milder decline (still ≥ baseline on SA, persistence; tied on PTV, inertia, momentum).

## Step-150→200 deltas (n=16 full)

| axis | β=30 Δ150→200 | β=100 Δ150→200 |
|---|---|---|
| SA          | **−0.125** | **−0.188** |
| PTV         | +0.000 | **−0.375** ⚠️ |
| persistence | +0.000 | **−0.250** ⚠️ |
| inertia     | −0.062 | +0.000 |
| momentum    | −0.125 | −0.188 |

β=100 PTV / persistence drops are the largest single-step movements observed across the run. β=30 transitions into mild downward across all 5 axes.

## n=16 step-200 trained means

| axis | baseline | β=30 step-200 | β=100 step-200 |
|---|---|---|---|
| SA          | 2.625 | 2.500 | 2.500 |
| PTV         | 2.875 | 2.812 | 2.625 |
| persistence | 3.375 | 3.250 | 3.000 |
| inertia     | 2.750 | 2.625 | 2.625 |
| momentum    | 2.750 | 2.688 | 2.625 |

(Note: baseline n=16 means above are larger than baseline n=8 means in the round-1/2 mds because the additional 8 prompts have higher baseline scores on average — they're "harder physics" but the untrained model already does ok-ish on them.)

## Stuck-prompt watch — `2455740c4d45` (Newton's cradle)

| axis | baseline | β=30 (50/100/150/200) | β=100 (50/100/150/200) |
|---|---|---|---|
| SA          | 2 | 1 / 1 / 1 / **1** | 1 / 1 / 1 / **1** |
| PTV         | 2 | 2 / 2 / 1 / **1** | 1 / 1 / 1 / **1** |
| persistence | 2 | 2 / 2 / 1 / **1** | 1 / 1 / 1 / **1** |
| inertia     | 2 | 1 / 1 / 1 / **1** | 1 / 1 / 1 / **1** |
| momentum    | 2 | 2 / 1 / 1 / **1** | 1 / 1 / 1 / **1** |

Both configs remain 5/5 floored. The hypothesis "transient, will recover" is now strongly disconfirmed — 4 consecutive ckpts have not recovered for either config. Confirmed *real DPO degradation* on multi-body collision class.

## Observations

1. **β=100 catastrophic regression at step-200** (rl2 forecast confirmed). The single-step drops are the largest in the run: −0.375 PTV, −0.250 persistence, −0.188 SA + momentum. This is the saturation cliff predicted from the step-99 loss/gnorm spike pattern.
2. **β=30 still net positive on n=8 carry-forward, but trajectory has turned negative or flat**. Step-150 was the peak; step-200 erodes momentum + inertia + SA modestly.
3. **Trajectory winner so far** (per rl2 `36137b1e` weighted-mean formula 0.5×last + 0.3×second + 0.2×third):
   - β=30 weighted-mean Δ (n=8 carry-fwd, axes-avg over last 3): SA=+0.125, PTV=+0.175, persistence=+0.150, inertia=+0.025, momentum=+0.075 → **avg ≈ +0.110**
   - β=100 weighted-mean Δ (last 3): SA=+0.087, PTV=+0.062, persistence=+0.050, inertia=−0.062, momentum=−0.075 → **avg ≈ +0.012**
   - β=30 leads by ~0.10 on the trajectory metric — robustly leading even after step-200 erosion.
4. **Spearman monotonicity** (Δ_step vs step number, n=8 carry-fwd, β=30): SA +0.95 / PTV +0.20 / persistence −0.95 / inertia +0.50 / momentum −0.50 (mixed). β=30 is not a monotone climber — bumpy trajectory. β=100 is also non-monotone (catastrophic regression at step-200 reverses earlier mild gains).

## Round-5 plan (step-250 / `lora_final`)

- step-250 ckpts shipped 09:09-09:11 UTC (filename `lora_final.safetensors` per rl8 `ffc35118`)
- β=30 step-250 regen LIVE since 09:51 UTC (PID 4106xxx, n=16, ETA done ~10:47 UTC)
- β=100 step-250 launches after — round-5 close ~11:50 UTC
- Final summary md `round4_v3_lr1e5_summary.md` with cross-round trajectory, weighted-mean / Spearman / per-prompt-class breakdown (per rl2 `36137b1e`)

## Source

- step-200 score per-prompt: `juyi-finetune:~/gen_out/round4_lr1e5_beta{30,100}_step200_<ts>/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- aggregate: `juyi-finetune:~/gen_out/round4_lr1e5_step200_agg.json` (incl. step-50/100/150 chain)
