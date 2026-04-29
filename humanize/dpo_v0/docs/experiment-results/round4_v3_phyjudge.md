# Round-4 v3: PhyJudge eval on tier_b 1k LoRA (β=1000, lora_rank=16, lr=5e-5, 200 step)

**Date**: 2026-04-29 (UTC)
**Status**: ❌ NOT ratified as round-4 final. Trained ckpt is downstream-degraded vs baseline on physical persistence + temporal coherence (sign-test p < 0.05).

## TL;DR

| signal | result |
|---|---|
| train metrics (acc_win50 / loss) | ✅ converged: acc_win50 = 0.88, loss < 0.1 in 82% of Q4 steps (rl1 wandb quartile pull) |
| PhyJudge persistence | ❌ trained −0.286 vs baseline (sign-test p = 0.007) |
| PhyJudge PTV (temporal coherence) | ❌ trained −0.190 vs baseline (sign-test p = 0.049) |
| PhyJudge SA / inertia / momentum | ≈ tied (high tie counts, no significant trend) |
| catastrophic regressions | 5 / 42 prompts lost −1 to −2 on 4-5 axes |

**Lesson**: high pair-preference acc_win50 ≠ high downstream video quality. Train metrics are necessary but not sufficient gate; Round-N ckpts must pass PhyJudge before being labeled a baseline.

## Training config (v3, FSDP)

- recipe `6bef6e104cdd3442` (canonical round-2 carryover)
- training config `b44cb5193ef8552b` — β=1000, lr=5e-5, lora_rank=16, lora_alpha=16, max_steps=200, max_pairs=1000, sigmoid loss
- pair_ids `cf5d3e5fd528a3e0` (round-4 1k subset of tier_b 2745)
- 8×A100 (juyi-videorl), `--dit-fsdp true`, grad-ckpt, sequential DPO (Hammer 2), ref-via-disabled-LoRA (Hammer 1)
- launch script: `script/launch_round4_train_v3_fsdp.sh`
- final ckpt (raw, FSDP-wrapped): `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T023311Z/lora_final.safetensors` (153 500 376 B, sha `18a5fd0be28ba0c7fedf6d295d84beb4268270f1a1c5508dc1aa9e44cd6ac689`)
- DiffSynth-stripped (`_fsdp_wrapped_module.` prefix removed, 800 keys): `juyi-finetune:~/m4_lora_gen_out/v3_round4/lora_final_stripped.safetensors` (153 440 328 B, sha `d97157a7c139dfbd3a2b7947687ec529fff2244c5ef9f4e1ec2118a30707f45a`)

## Train trajectory (rl1 wandb quartile pull)

Repro command:

```bash
python humanize/dpo_v0/eval/analyze_wandb_dpo_run.py \
  --run-path lukelin/wanrl/9if26kr9 \
  --token-path /shared/user60/worldmodel/rlvideo/videodpoWan/wandbtoken
```

Metric read: W&B `accuracy` is the per-step sign bit emitted by
`train_dpo_i2v.py` (`1.0 if margin/logit > 0 else 0.0`). It is expected to be
noisy on raw step plots. `acc_win50` is the trainer-side rolling 50-step mean
and is the metric used here for convergence interpretation.

| Quarter | acc_win50 | margin mean | loss mean | loss<0.1 frac |
|---|---|---|---|---|
| Q1 (0-49) | 0.57 | +2.22 | 1.69 | 0.26 |
| Q2 (50-99) | **0.48** | **−1.82** | **3.86** | 0.24 |
| Q3 (100-149) | 0.69 | +15.33 | 0.28 | 0.68 |
| Q4 (150-199) | **0.88** | +16.30 | 0.27 | **0.82** |

U-shape: real Q2 unlearning trough → strong Q3-Q4 recovery. Initially mistaken for "stuck" by mid-run snapshot reading; quartile aggregate showed the recovery was real.

**However**: this U-shape is exactly the dynamic that produced an over-fit to pair-preference noise, which only shows up downstream at PhyJudge. β=1000 + rank=16 is a large enough capacity / sigmoid steepness combo to push policy across the boundary on a small 1k tier_b subset.

## Eval setup

- **Heldout**: 42 prompts × 2 modes (baseline = no LoRA, trained = v3 LoRA), byte-identical generation_config (gen_config_sha256 = `e31262d26c7c3cdeccf2ace12183e3021792dfb1219f68e7658c067348284288`).
- **Generator**: `humanize/dpo_v0/eval/heldout_regen.py` with `--cache-pipe --mode-batched`, juyi-videorl 8-rank, ~14 min/video gen wall, ~2h39m total wall (16:29–19:08 UTC).
- **Output**: 84 videos at `juyi-videorl:~/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z/heldout_regen/<prompt_id>/<mode>/<mode>/<ts>/{video.mp4, manifest.json}`. Tar-piped (37 MB) → juyi-finetune for scoring.
- **Scorer**: `videodpo:benchmark/physground_score.py` (rl8 #40 refactor: vision-cache + summary + max-jobs + ETA). Run on juyi-finetune `physground` env, 5 axes (SA, PTV, persistence, inertia, momentum), ~13 min wall (84 videos × 5 axes × ~9.4 s/video avg).
- **Score artifact**: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/{summary.json, results.jsonl, <run_id>.json × 84}`.

## Pre-eval bug & fix

First eval attempt produced **all-zero deltas** because trained-mode LoRA never attached: keys in the trainer-saved safetensors had `_fsdp_wrapped_module.` prefix from FSDP wrapping. DiffSynth's `pipe.load_lora` silently no-op'd on the unmatched keys.

- Detection: `sha256(baseline_video) == sha256(trained_video)` for all 8 smoke prompts (rl8 `9d60639c`).
- Fix (inference-side, immediate): `strip_fsdp_prefix.py` writes a cleaned `lora_final_stripped.safetensors` (rl7).
- Fix (trainer-side, permanent): `train_dpo_i2v.py:collect_lora_state` strips `_fsdp_wrapped_module.` segments before save (rl9 task #42 commit `2d44f08`). Future ckpts no longer need the manual strip.

A separate orchestrator-side race on `_mw_atomic_write_json(run_dir/run_manifest.pre.json)` was also fixed (rl7 — rank 0 owns canonical `run_manifest.json`, other ranks write per-rank `run_manifest.rankN.json`).

## Result tables

### Per-axis pooled mean (n=42 pairs, 1-5 integer Likert)

| axis | baseline mean ± std | trained mean ± std | mean Δ | sign-test p (two-tailed exact binomial) |
|---|---|---|---|---|
| SA (prompt alignment) | 2.690 ± 0.715 | 2.667 ± 0.902 | −0.024 | 1.0000 |
| PTV (temporal coherence) | 2.952 ± 0.731 | 2.762 ± 0.932 | **−0.190** | **0.0490** ⚠️ |
| persistence | 3.357 ± 0.879 | 3.071 ± 0.973 | **−0.286** | **0.0074** ⚠️ |
| inertia | 2.762 ± 0.692 | 2.714 ± 0.864 | −0.048 | 1.0000 |
| momentum | 2.833 ± 0.730 | 2.714 ± 0.835 | −0.119 | 0.5488 |

### Win/loss/tie counts per axis (42 paired prompts)

| axis | + wins (trained > baseline) | − losses (trained < baseline) | = ties |
|---|---|---|---|
| SA | 6 | 7 | 29 |
| PTV | 4 | 13 | 25 |
| persistence | 2 | 13 | 27 |
| inertia | 4 | 5 | 33 |
| momentum | 4 | 7 | 31 |

### Catastrophic regressions (worst 5 prompts)

| prompt_id | SA | PTV | persistence | inertia | momentum |
|---|---|---|---|---|---|
| 36e42af19937 | −2 | −1 | −2 | −2 | −2 |
| e90b3f54bffb | −1 | −1 | −1 | −1 | −1 |
| 2455740c4d45 | −1 | −1 | −1 | −1 | −1 |
| 7977e8df650c |  0 | −1 | −1 | −1 | −2 |
| 48255a441729 | −1 | −1 | −1 |  0 | −1 |

### Best wins (top 3 prompts where trained beat baseline)

| prompt_id | SA | PTV | persistence | inertia | momentum |
|---|---|---|---|---|---|
| 242e01f46c08 | +1 | +1 | +1 | +1 | +1 |
| e38a4396df92 | +1 | +1 |  0 | +1 | +1 |
| 70d3b1b89e19 | +2 | +2 |  0 |  0 |  0 |

## Diagnosis

Two directions consistent with the data:

1. **Training config too aggressive**: β=1000 + lora_rank=16 + lr=5e-5 is enough capacity / sigmoid steepness for the policy to over-fit pair-preference ranking on 1k tier_b without recovering global video-quality structure. Persistence + PTV (the axes most sensitive to whole-clip coherence) take the biggest hit.
2. **tier_b 1k pair noise**: pair winner labels may not perfectly align with PhyJudge axes (DB rating signal vs. visual judge). Even an "optimal" LoRA fit to noisy pair labels would degrade on PhyJudge.

These are not mutually exclusive. To disambiguate, the cleanest experiment is to PhyJudge-score the round-2 M4 ckpt (200-pair tier_b, 50-step, lr=1e-5, β=0.1, lora_rank=8 — much more conservative training config on the same dataset family). Path:

- M4 ckpt: `juyi-finetune:~/m4_ckpt/lora_final_diffsynth.safetensors` (DiffSynth-converted, `1ea81964…`)
- Reuse v3 baseline videos via `--baseline-from <prior_run_dir>` (rl7 landed flag in `heldout_regen.py`); ~45 min wall on juyi-finetune 4-rank for trained-only pass.

If M4 also regresses → pair noise dominates, action is on dataset side.
If M4 holds or wins → training config dominates, action is Option A (β=100, lora_rank=8) re-train.

## Round-5 spec implications

- **Always PhyJudge before ratifying any ckpt as baseline.** Train metrics are gate input, not gate output.
- `inference_steps` is locked at 50 (luke `6aff43d1`); do not change for cross-round comparability.
- `--baseline-from <prior_run_dir>` reuse is now landed; round-5 trained-only pass is ~50% wall save vs full pair regen.
- Multi-box parallel inference (juyi-videorl 8 + juyi-finetune 4) is a follow-up wall reduction; rl7 + rl1 to design.
- `Option A` config (β=100, lora_rank=8, lr=5e-5, 200 step, FSDP) and `smokeD` (β=0.1, lora_rank=8, lr=1e-5, max_steps=1) are already pinned and pushed to `origin/main` (commit `30b6a9d`); pins `e9380254b3e39385` / `9f1aafc8ddc66884`. Launch scripts in `script/launch_round4_optionA.sh` + `script/launch_round4_smokeD.sh` (gitignored, on shared FS / `~/videodpoWan-task20/`).

## Key artifacts

| artifact | path |
|---|---|
| v3 ckpt (raw, FSDP-prefixed) | `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T023311Z/lora_final.safetensors` (sha `18a5fd0be28b…`) |
| v3 ckpt (DiffSynth-stripped) | `juyi-finetune:~/m4_lora_gen_out/v3_round4/lora_final_stripped.safetensors` (sha `d97157a7c139…`) |
| 84 generated videos | `juyi-videorl:~/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z/heldout_regen/<prompt_id>/{baseline,trained}/<mode>/<ts>/video.mp4` (also tar-piped to juyi-finetune at same relative path) |
| PhyJudge per-video JSON | `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/<run_id>.json` (84 files) |
| PhyJudge results.jsonl | `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/results.jsonl` |
| PhyJudge summary.json | `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/summary.json` |
| Scorer source | `videodpo:benchmark/physground_score.py` @ commit `5fcdcc2` (#40, vision-cache + summary + max-jobs + ETA) |

## Co-discovery credits

- v3 training launch + monitor + final summary fetch: rl1
- #39 control train (lr=1e-5) + train metrics interp: rl7 + rl1
- Eval generator + cache_pipe + race fix + FSDP-prefix strip: rl7
- Trainer-side FSDP-prefix strip fix (round-N+ no-manual-strip): rl9 task #42 commit `2d44f08`
- Scorer optimizations + per-axis paired-delta + sign-test analysis + bug detection (byte-equal trained=baseline): rl8 task #40
- Dispatcher coordination + retracted Option A push (then re-endorsed when downstream data flipped the read): rl2
