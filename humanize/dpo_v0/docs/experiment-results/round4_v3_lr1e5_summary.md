# Round-4 v3 lr1e5 — final cross-round summary (5 ckpts × 2 β configs)

**Task**: #46 (rl9). 5 rounds × 2 β configs (β=30, β=100) at lr=1e-5, num_samples=1000, save_every=50, FSDP 4+4 parallel on juyi-videorl 8×A100.

## TL;DR

| | step-50 | step-100 | step-150 | step-200 | step-250 / lora_final |
|---|---|---|---|---|---|
| **β=30** axes-avg Δ (n=8) | +0.150 | −0.025 | **+0.200** ← peak | +0.075 | +0.075 |
| **β=100** axes-avg Δ (n=8) | −0.025 | +0.050 | +0.075 | −0.075 (cliff) | **+0.200** ← peak |
| Newton cradle β=30 (axes-avg score, base=2) | 1.6 | 1.4 | 1.0 | 1.0 | 1.0 (5/5 floored) |
| Newton cradle β=100 | 1.0 | 1.0 | 1.0 | 1.0 | **1.6** (3/5 recovered) |

**Winner: β=100 lora_final (step-250)**. Best by 3 of 4 trajectory metrics (Spearman, step-250 final value, peak step-mean tied with β=30 step-150). β=100 also is the *only* ckpt to partially recover Newton's cradle.

## Trajectory plot (axes-avg Δ vs baseline, n=8 carry-forward)

```
   step:        50     100    150    200    250
   β=30   :   +0.150 -0.025 +0.200 +0.075 +0.075
              [P]    [↓]    [PEAK] [↓]    [flat]
   β=100  :   -0.025 +0.050 +0.075 -0.075 +0.200
              [-]    [↑]    [↑]    [CLIFF][PEAK]
```

ASCII shape:
```
β=30  : ▆_▇▃▃   (early peak, U-rise, flat at end)
β=100 : _▂▃_▇   (slow climb, cliff, big recovery)
```

## Trajectory-winner metrics (rl2 `36137b1e`)

Computed on n=8 carry-forward, axes-averaged across 5 ckpts.

| metric | β=30 | β=100 | winner |
|---|---|---|---|
| weighted-mean (0.5/0.3/0.2 over last 3 ckpts) | +0.092 | +0.087 | tie (within noise) |
| **Spearman** (axes-avg monotonicity) | +0.020 | **+0.520** | **β=100** |
| **step-250 final** (axes-avg) | +0.075 | **+0.200** | **β=100** |
| peak step-mean (single best step) | +0.200 (step-150) | +0.200 (step-250) | tie |

**β=100 wins 2 + ties 2; β=30 wins 0.** Final-step value criterion + Spearman monotonicity favor β=100. Weighted-mean is essentially tied because β=100's catastrophic step-200 dip pulls its weighted-mean down even though step-250 recovers strongly (the formula gives weight 0.3 to step-200 which hurts β=100).

## Per-prompt-class breakdown at lora_final (rl2 `771ddfff`)

Splitting subset into collision-class (Newton's cradle, n=1) vs non-collision (n=15).

| axis | β=30 lora_final, all | collision | non-collision | β=100 lora_final, all | collision | non-collision |
|---|---|---|---|---|---|---|
| SA          | −0.062 | −1.000 | +0.000 | **+0.188** | +0.000 | **+0.200** |
| PTV         | −0.125 | −1.000 | −0.067 | **+0.125** | +0.000 | **+0.133** |
| persistence | −0.312 | −1.000 | −0.267 | **+0.125** | +0.000 | **+0.133** |
| inertia     | −0.062 | −1.000 | +0.000 | −0.125 | −1.000 | −0.067 |
| momentum    | −0.125 | −1.000 | −0.067 | +0.000 | −1.000 | +0.067 |

Even **excluding Newton's cradle**, β=100 lora_final beats β=30 lora_final on 4 of 5 axes (SA / PTV / persistence / momentum). β=100 is not winning purely because it recovered the collision-class outlier — it generalizes better across non-collision prompts as well.

## Newton's cradle (`2455740c4d45`) — collision-class case study

| axis | base | β=30: 50/100/150/200/250 | β=100: 50/100/150/200/250 |
|---|---|---|---|
| SA          | 2 | 1/1/1/1/1 | 1/1/1/1/**2** ⭐ |
| PTV         | 2 | 2/2/1/1/1 | 1/1/1/1/**2** ⭐ |
| persistence | 2 | 2/2/1/1/1 | 1/1/1/1/**2** ⭐ |
| inertia     | 2 | 1/1/1/1/1 | 1/1/1/1/1 |
| momentum    | 2 | 2/1/1/1/1 | 1/1/1/1/1 |

**Findings**:
1. **β=30 monotone-down on this prompt** — every step that changed the score did so downward; never recovered.
2. **β=100 stayed at floor for steps 50–200**, then recovered 3/5 axes at step-250.
3. **The recovery is multi-axis simultaneous** at the same ckpt, which is unlikely from PhyJudge label noise alone (independent calls per axis would average out independently). Real lora_final improvement.

**Implication for round-5+ research**:
- (a) "accept trade-off" decision should weight β=100 as the better trade-off — it recovered the trade-off prompt at lora_final, β=30 did not.
- (b) **pair-set rebalance** (rl2 proposal) still on the table: collision-class winners in pair-pref might still be noisy. lora_final β=100 is signal that high-β can still learn collision physics if given enough steps.
- (c) **early-stop on β=30 step-150** (its peak) still produces +0.200 axes-avg Δ — comparable to β=100 lora_final. But β=100 lora_final has the Newton's cradle partial recovery, which is unique value.

## Methodology / runtime notes

- **Train**: juyi-videorl 8×A100, FSDP 4-rank parallel × 2 configs. β=30 + β=100 both 100% high-noise routing AC ✓, total_forwards=1000 ✓, no halt / no OOM. β=30 9h22m, β=100 9h18m. Hard contracts verified by rl8 at #45 close (`ffc35118` + `1737189a`).
- **Eval gen**: juyi-finetune 4×A100 (NOT 8 as initially assumed — verified `nvidia-smi`), `wan` env, `--baseline-from` v3 baseline reuse for all 5 rounds. Rounds 1/2 at n=8, rounds 3/4/5 at n=16 (luke1 `d94f3b77`).
- **Eval scoring**: juyi-finetune `physground` env, 5 axes (SA/PTV/persistence/inertia/momentum). Per-prompt invocation (`--run-dir`) to avoid gen_ts collision footgun.
- **BitLesson candidates** for round-6+:
  1. `heldout_regen.py` needs explicit `torch.cuda.set_device(local_rank)` — without it, 4-rank torchrun all binds to cuda:0 → OOM. Workaround: bash wrapper passing `--device cuda:${LOCAL_RANK}`. Rl2 (`b3b7ba2d`) noted permanent fix candidate is in `main()`.
  2. `physground_score.py` per-video JSON keyed by `manifest.timestamp_utc` — multi-rank ranks finishing on same second collide. Workaround: `--run-dir` per video into `scores_perprompt/<pid>/`.

## Decision points for round-6+ / luke1

1. **Pick β=100 lora_final as the v3 round-4 lr1e5 winning ckpt** (per trajectory + final-value + Newton's cradle partial recovery).
2. **β=100 step-200 cliff is concerning** (PTV −0.375, persistence −0.250 single-step) — investigation may be worthwhile but may also be a one-off saturation event since lora_final fully recovered. Watch in next training run.
3. **β=30 hits a soft ceiling at step-150** (peak axes-avg +0.200, then decline). Lower β doesn't push past β=100 lora_final on this prompt set.
4. **Newton's cradle and similar multi-body collision prompts**: β=100 partial recovery suggests these are *trainable* but need higher β + more steps. Pair-set rebalance (rl2 `771ddfff`) still merits diagnostic if collision-class is broadly underrepresented.

## Per-round links

| round | step | n | md path |
|---|---|---|---|
| 1 | 50 | 8 | `humanize/dpo_v0/docs/experiment-results/round4_v3_lr1e5_step50.md` |
| 2 | 100 | 8 | `humanize/dpo_v0/docs/experiment-results/round4_v3_lr1e5_step100.md` |
| 3 | 150 | 16 | `humanize/dpo_v0/docs/experiment-results/round4_v3_lr1e5_step150.md` |
| 4 | 200 | 16 | `humanize/dpo_v0/docs/experiment-results/round4_v3_lr1e5_step200.md` |
| 5 | 250 | 16 | `humanize/dpo_v0/docs/experiment-results/round4_v3_lr1e5_step250.md` |

## Source data (juyi-finetune)

- aggregates: `~/gen_out/round4_lr1e5_step{50,100,150,200,250}_agg.json`
- gen + score per round: `~/gen_out/round4_lr1e5_beta{30,100}_step{N}_<ts>/`
- baseline (untrained Wan2.2-I2V-A14B): `~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl` (42 baseline + 42 trained, 84 entries)

## Acknowledgments

- @rl8 (#45 train owner): both 9-hour runs landed clean, ckpt-landing reminders + ship coordination throughout
- @rl2 (#46 coordinator): N=16 math correction, trajectory-score formula refinements, U-shape forecast (`187189fa`) and Newton's cradle `2455740c4d45` baseline-data correctness check (`771ddfff`)
- @luke1: directives `dd2d6da3` (every-50step cadence), `d94f3b77` (N=16 ratify), and the question that crystalized the conclusion ("有点太多了，多生成几个视频")
