# Round-4 v3 lr1e5 step-50 PhyJudge eval

**Task**: #46 (rl9). Round 1 of 5 — luke1 directive `dd2d6da3` "之后每 50step 你都要 eval".

## Run identity

| field | β=30 | β=100 |
|---|---|---|
| LoRA ckpt | `lora_step50.safetensors` (β=30, 153.4 MB) | `lora_step50.safetensors` (β=100, 153.4 MB) |
| ckpt sha256 | `04b2f15b9fc6…` | (queried at score time, see results.jsonl) |
| ckpt path on juyi-videorl | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234923Z/` | `~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234925Z/` |
| trainer commit | `8bde87b` (rl9 #42 fsdp-strip applied at save → ckpts ship-clean) |
| training | juyi-videorl 4-rank parallel (GPU 0-3 β=30, GPU 4-7 β=100), num_samples=1000, target_steps=250, save_every=50; per-step ~133 s |
| eval gen | juyi-finetune 4-rank, `--limit-prompts 8`, `--baseline-from` v3 baseline reuse, `multi_gpu_inference_seed_parallel` envelope; per-rank wrapper sets `--device cuda:${LOCAL_RANK}` (heldout_regen.py default `cuda` collides on rank-0 OOM at 4-rank) |
| eval scoring | physground env on juyi-finetune, 5 axes (SA, PTV, persistence, inertia, momentum), `--run-dir` per video to avoid gen_ts collision |
| eval out | `juyi-finetune:~/gen_out/round4_lr1e5_beta{30,100}_step50_<ts>/` (regen) + `.../scores_perprompt/<pid>/<run_ts>/` (score) |

## Subset (8 prompts, deterministic by `prompt_id` first-alpha)

| prompt_id | caption |
|---|---|
| `1a0d4f1d8b1a` | Two players reach for a volley, their racquets meet, creating a visible impact. |
| `1a44aba35343` | A person carries a heavy bucket while wading through chest-deep water. |
| `1b1c06c5ff1c` | A 30lb kettlebell is slowly lowered on top of a yellow ceramic coffee mug placed on a wooden table. The ceramic mug shatters under the weight of the kettlebell. Static shot with no camera movement. |
| `242e01f46c08` | A large pumpkin is placed on a small, flimsy stool; the stool breaks, and the pumpkin rolls onto the ground. |
| `2455740c4d45` | A Newton's cradle device on the table and one of the metal balls is held up by a blue handled grabber tool. The claw releases the ball. The released ball swings down, hits the stationary balls, and the ball on the opposite end swings out. Static shot with no camera movement. |
| `24d86e4e0339` | A bowling ball rolls down a polished wooden lane, hitting the pins at the end. |
| `252b84def499` | The person gently places an egg yolk in a measuring cup into a pot of boiling water. The egg yolk quickly cooks and solidifies due to the heat. |
| `2559ab47b909` | A player uses a backhand shot to send a racquetball into the corner, the ball rebounding off two walls. |

## Results — score means (n=8, integer 1–4 PhyJudge scale)

| axis | v3 baseline | β=30 step-50 | β=100 step-50 |
|---|---|---|---|
| **SA**          | 2.375 ± 0.518 | 2.375 ± 0.744 | 2.250 ± 0.707 |
| **PTV**         | 2.625 ± 0.744 | 2.750 ± 0.707 | 2.500 ± 0.756 |
| **persistence** | 2.875 ± 0.991 | 3.125 ± 0.835 | 3.000 ± 1.195 |
| **inertia**     | 2.500 ± 0.535 | 2.625 ± 0.744 | 2.500 ± 0.756 |
| **momentum**    | 2.500 ± 0.535 | 2.750 ± 0.463 | 2.625 ± 0.744 |

## Δ vs baseline (trained − baseline, per-prompt then mean)

| axis | β=30 step-50 Δ | β=100 step-50 Δ |
|---|---|---|
| **SA**          | +0.000 ± 0.535 | −0.125 ± 0.354 |
| **PTV**         | +0.125 ± 0.641 | −0.125 ± 0.641 |
| **persistence** | +0.250 ± 0.463 | +0.125 ± 0.641 |
| **inertia**     | +0.125 ± 0.641 | +0.000 ± 0.535 |
| **momentum**    | +0.250 ± 0.463 | +0.125 ± 0.641 |

### Per-prompt Δ (debug)

prompt order matches subset table above.

| axis | β=30 Δ per prompt | β=100 Δ per prompt |
|---|---|---|
| SA          | `[0, 0, 0, +1, −1, 0, 0, 0]` | `[0, 0, 0, 0, −1, 0, 0, 0]` |
| PTV         | `[+1, +1, −1, 0, 0, 0, 0, 0]` | `[+1, 0, −1, 0, −1, 0, 0, 0]` |
| persistence | `[+1, 0, 0, +1, 0, 0, 0, 0]` | `[+1, 0, 0, 0, −1, 0, 0, +1]` |
| inertia     | `[+1, 0, 0, +1, −1, 0, 0, 0]` | `[+1, 0, 0, 0, −1, 0, 0, 0]` |
| momentum    | `[+1, 0, 0, +1, 0, 0, 0, 0]` | `[+1, 0, 0, +1, −1, 0, 0, 0]` |

## Observations

1. **β=30 dominates β=100 at step-50.** β=30 mean Δ is ≥0 across all 5 axes, with persistence (+0.250) and momentum (+0.250) the strongest. β=100 shows mild regression on SA (−0.125) and PTV (−0.125), and no improvement on inertia.
2. **Highest-impact axes** for β=30 are *persistence* (+0.250) and *momentum* (+0.250) — both physical-continuity axes (object remains itself / motion stays consistent). PhyJudge picks these up first because they degrade most visibly when the model overshoots its base prior; DPO at β=30 is enough to recover them at step-50.
3. **Prompt `2455740c4d45` (Newton's cradle)** is the worst prompt at step-50 for *both* configs — it regresses on SA / PTV / persistence / inertia / momentum under β=100, and on SA / inertia under β=30. This suggests the model is not yet aligned on the precise multi-body collision constraint at 50 steps; expect step-100/150 to recover it (or flag it as a stuck-prompt for round-5).
4. **Prompt `1a0d4f1d8b1a` (volley impact)** is the strongest β=30 win — +1 on PTV / persistence / inertia / momentum (no SA gain). Same prompt at β=100: only +1 on PTV / persistence / inertia / momentum (matches β=30, which means the prompt is broadly responsive).
5. **Sample size n=8 is too small for statistical confidence**; the per-axis means here are point estimates with std ≈ 0.5–0.7 score units, so the Δ ≈ 0.125 differences are within noise. Use them only as a *direction-of-travel* signal across the rolling step-50/100/150/200/250 cadence.

## Caveats

- **Subset only**, full 42-prompt eval is not run per luke1's 1h17m budget directive `d78dd892`. Round-5 should pick the trend-confirming step (peak Δ across all axes before regression) and run the full 42 there.
- **PhyJudge is a discrete 1–4 scale**; Δ is integer-quantized so means like +0.125 mean only "1 of 8 prompts moved by 1, the rest unchanged". Do not interpret as continuous improvement.
- **`--baseline-from`** reuses the v3 baseline videos (untrained Wan2.2-I2V-A14B + same gen_config). gen_config_sha256 is asserted byte-equal between current run and the v3 baseline run; if it ever drifts, regen halts (AC-7.2). Confirmed equal at round-1.
- **Score script gen_ts collision**: physground_score per-video JSON is keyed by `manifest.timestamp_utc`; with 4 ranks finishing on the same second some files get overwritten. Workaround: invoke score with `--run-dir` per video into a `scores_perprompt/<pid>/` tree. Filed as candidate BitLesson `physground-score-gen-ts-collision`.

## Round-2 plan (step-100, in flight)

| ckpt | regen launched | ETA |
|---|---|---|
| β=30 step-100 | 03:39 UTC | 04:07 UTC |
| β=100 step-100 | (after β=30) | 04:35 UTC |
| score 16 videos | 04:35 UTC | 04:42 UTC |
| `round4_v3_lr1e5_step100.md` | — | 04:45 UTC |

Step-150 ckpts ETA ~05:26 UTC → 41 min buffer between round-2 close and round-3 kickoff.

## Source data

- baseline scores: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl` (entries with `/baseline/baseline/` in path, 8 of 42 selected)
- β=30 step-50 scores: `juyi-finetune:~/gen_out/round4_lr1e5_beta30_step50_20260430T022000Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- β=100 step-50 scores: `juyi-finetune:~/gen_out/round4_lr1e5_beta100_step50_20260430T025236Z/scores_perprompt/<pid>/<run_ts>/results.jsonl`
- aggregate JSON: `juyi-finetune:~/gen_out/round4_lr1e5_step50_agg.json`
