# Round-4 Summary — Wan2.2-I2V-A14B Direct-I2V DPO (1k tier_b @ β=1000 lr=5e-5)

Completion: 2026-04-29 12:50 UTC. Round-4 closed without ratifiable ckpt.

## TL;DR

- v3 training **converged** on the pair-preference task: final `acc_win50=0.88`, `loss<0.1`占 82% of Q4 steps.
- v3 ckpt PhyJudge 42-prompt eval **regressed against baseline** on 2 of 5 axes:
  - persistence: Δ = **−0.286** (sign-test p = **0.007**, 13 losses / 2 wins / 27 ties out of 42)
  - PTV (temporal coherence): Δ = **−0.190** (p = **0.049**, 13 losses / 4 wins / 25 ties)
  - SA / inertia / momentum: directionally negative but not significant
- 5/42 prompts catastrophically regressed (−1 to −2 across 4–5 axes).
- **train metric ≠ video quality** is the headline lesson: high `acc_win50` on pair-pref is necessary but not sufficient for downstream PhyJudge.
- v3 ckpt **NOT ratified** as round-4 final. Round-5 enters from the question: was this hurt by training config (β=1000 / lora_rank=16) or by tier_b pair noise leaking through preference signal?

## Provenance

### Decisions (lukedecision.md, 2026-04-28)
- A2: tier_b scope-up to ~1000 pair (not full 2745).
- A1#5: lr=5e-5, 200-step proof-of-method.
- B1 DEC-CAL relax: loss target ≤0.30 not enforced; halt only on routing/divergent.
- B2: only round-4 M4-tier ckpt evaluated (skip M3).
- C3: v8 trainer 3 invariants documented as round-4+ baseline anchor.

### Round-4 r5 config delta (commit `257a828`)
- `training_config_round4.yaml`: `beta: 0.1 → 1000` ("diffusion-DPO scale; r4 saturated at ln(2)") with rationale that round-2's stagnation at `ln(2)` was a signal-too-dim symptom. Final analysis (this round) shows the stagnation root cause was `lr=1e-5 + 50 step` being too short for movement, not β too small. β=1000 induced asymmetric sigmoid saturation: tiny correct margin → instant vanishing gradient; tiny wrong margin → full β-scaled gradient. Combined with `lr=5e-5 + 200 step` it pushed LoRA into a regime where pair-pref accuracy compounded but downstream physics failed.

### Three pins (round-4)
| Pin | Value | Source |
|-----|-------|--------|
| recipe_id (data preprocessing canonical) | `6bef6e104cdd3442` | immutable carryforward from round-2 |
| pair_ids_sha256_hex16 (1k subset, newline-canonical) | `cf5d3e5fd528a3e0` | `humanize/dpo_v0/build_round4_tier_b_1k.py` |
| training_config_sha256_hex16 (knobs canonical) | `b44cb5193ef8552b` (r5, β=1000) | r4 was `06d338945115dcc3` (β=0.1) |

## Pipeline (commit chain)

| Commit | Purpose | Author |
|--------|---------|--------|
| `c505e09` | task-19 r1: subset builder (2745 → 2202 cond-image-present → 1k seed-shuffled) | rl1 |
| `56164a8` | task-19 r2: pair_ids canonical sha256 pin (newline-joined form, byte-equal to trainer assert) | rl1 |
| `1c35ec9` | task-19 r3: training_config double-pin (recipe_id + training_config_sha256) | rl1 |
| `7970ca8` | task-20 r1: trainer round-4 dual-pin asserts + 12 new tests | rl1 |
| `a968fc1` | task-20 r2: encode_videos.py `--tier tier_b_round4_1k` mode | rl1 |
| `54a90b7` | task-20 r3: encode_videos.py env-var fallback (deploy hygiene for `/shared` cross-box) | rl1 |
| `cc556db` | task-30 r1: pipe-cache + mode-batched (closure caches WanVideoPipeline + LoRA toggle, per-rank: baseline batch → attach LoRA → trained batch = 1 build + 1 attach instead of N×builds) | rl7 |
| `257a828` | round-4 r5: β 0.1 → 1000 | luke |
| `9ba517f` | grad clip + reward / margin / accuracy / acc_win50 metrics | rl1 |
| `2d44f08` | task-42: collect_lora_state strips `_fsdp_wrapped_module.` segments at save time (trainer-side fix; supersedes round-4 manual strip workaround) | rl9 |

## v3 Training Run

### Config
- box: juyi-videorl 8×A100 80GB
- script: `script/launch_round4_train_v3_fsdp.sh`
- FSDP: `--dit-fsdp true` (FULL_SHARD with use_orig_params=True; per-rank base ~3.5 GB)
- env: `wan` conda (torch 2.5.1+cu121, flash_attn 2.7.3, torchvision 0.20.1, diffsynth)
- knobs: lr=5e-5, β=1000, lora_rank=16, lora_alpha=16, max_steps=200, max_pairs=1000, micro_batch=1, sampling_band=[901,999]
- v8 invariants preserved (Hammer 1 ref-via-disabled-LoRA + Hammer 2 sequential DPO + Hammer 3 grad-ckpt monkey-patch)
- run_dir: `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T023311Z/`

### Wall + memory
- 200 step on 8-rank FSDP DDP, ~134 s/step, total wall ~7h28m (from 02:33 UTC start to 10:01 UTC done)
- VRAM peak ~62 GB / 80 GB cap, ~22% margin (above v8 10% min)
- grad_finite=1 全程, no NaN

### Train metrics (rl1 wandb quartile fetch)
| Quarter | acc_win50 | margin mean | loss mean | loss<0.1 frac |
|---|---|---|---|---|
| Q1 (0–49)   | 0.57 | +2.22  | 1.69 | 0.26 |
| Q2 (50–99)  | **0.48** | **−1.82** | **3.86** | 0.24 |
| Q3 (100–149)| 0.69 | +15.33 | 0.28 | 0.68 |
| Q4 (150–199)| **0.88** | +16.30 | 0.27 | **0.82** |
- U-shape: Q2 unlearning trough → Q3-Q4 strong recovery + convergence on pair-pref task.
- Mid-run snapshot at step 34 (`acc_win50=0.57`) was misread as "stuck" (rl2 + rl7 premature amber) before final summary fetch corrected.

Repro:
```bash
python humanize/dpo_v0/eval/analyze_wandb_dpo_run.py \
  --run-path lukelin/wanrl/9if26kr9 \
  --token-path /shared/user60/worldmodel/rlvideo/videodpoWan/wandbtoken
```

Metric read: W&B `accuracy` is the per-step sign bit emitted by `train_dpo_i2v.py` (`1.0 if margin/logit > 0 else 0.0`) and is expected to be noisy on raw step plots. `acc_win50` is the trainer-side rolling 50-step mean and is the metric used here for convergence interpretation.

### Lesson #1: monitor pipeline must fetch final summary
- High-β DPO has expected mid-run unlearning trough due to asymmetric sigmoid saturation.
- 9-step samples or step-N snapshots miss U-shape recovery.
- Quartile / rolling aggregate views are required for amber → red-light decisions during training.

## #39 Control (lr=1e-5, β=1000, lora_rank=16) — discriminator

- box: juyi-finetune 4×A100 80GB
- run_dir: `juyi-finetune:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T042438Z/`
- final `acc_win50=0.56` (vs v3's 0.88), margin mean stayed in ±1 throughout (never activated β=1000 sigmoid flip), loss<0.1 occupancy 4–20% across quarters.

### Lesson #2: lr controls activation; β controls sharpness
- lr=1e-5 + β=1000 = stuck in neutral zone (Δ ≈ 0 throughout). β=1000's chosen sigmoid steepness is wasted when policy can't generate margin.
- v3 lr=5e-5 + β=1000 was the regime that activated the sharp sigmoid AND drove margins large.
- (#39 PhyJudge eval was started in parallel at #41 but stopped on luke's "全停" directive at 12:54; partial 24/84 baseline videos retained but not scored.)

## v3 PhyJudge eval pipeline (#41 + #40)

### Critical bug 1: FSDP-wrapped LoRA save format
- v3 ckpt at `lora_final.safetensors` had keys with `_fsdp_wrapped_module.` prefix segments (multi-level FSDP wrap).
- DiffSynth `pipe.load_lora` matched against raw `pipe.dit` keys → silent no-op load → trained mode produced byte-equal output to baseline → 4-prompt smoke initial run all-zero deltas.
- **Caught**: rl8 score smoke saw `mean Δ = 0.0` everywhere → flagged "scorer correct, pipeline silently broken" → rl7 traced to `attach_lora` log lines showing call but no key matches → strip workaround (`strip_fsdp_prefix.py`) renames `_fsdp_wrapped_module.` → `""` to restore DiffSynth-compatible keys. 800 LoRA tensors, 153 MB, weight bytes intact.
- **Permanent fix**: rl9 task #42 (`2d44f08`) — `collect_lora_state` walks model with `name.replace("_fsdp_wrapped_module.", "")` before generating state keys. Round-5+ ckpts will save in DiffSynth-native format directly.
- **Verification**: smoke 4-prompt with stripped ckpt showed trained ≠ baseline at byte and score level (per-prompt up to ±1 swing on integer scale).

### Critical bug 2: race in atomic_write_json
- `regen_all` had each rank write `run_dir/run_manifest.pre.json` via `_mw_atomic_write_json`. 4 ranks racing on same tmp file → crash at ~step 4/8 in v3 strip smoke.
- **Fix** (rl7, in shared `heldout_regen.py`): rank 0 writes marker `.run_ts` and canonical `run_manifest.json`. Other ranks poll for marker and use the same UTC dir, then write per-rank `run_manifest.rank{N}.json` (no contention). 40/40 heldout_regen tests pass.

### Eval setup
- 42 unique heldout prompts × 2 modes (baseline + trained) = 84 video generations
- byte-identical generation_config (AC-7.2) verified at startup; `gen_config_sha256 = e31262d26c7c3cdeccf2ace12183e3021792dfb1219f68e7658c067348284288`
- inference_steps=50 (locked), seed=42, sampler=uni_pc, resolution=832x480, num_frames=81
- 8-rank seed-parallel on juyi-videorl 8×A100 with `--cache-pipe --mode-batched` (per rank: 1 build + 1 attach + N gens)
- per-video wall ~14 min (50 step × ~17s/step + VAE encode/decode + save), total wall ~2h39m
- run_dir: `juyi-videorl:~/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z/heldout_regen/`

### Score (rl8 #40 scorer at `videodpo:5fcdcc2`)
- Vision-cache scorer on juyi-finetune `physground` env, 5 axes
- 84 videos × 5 axes × ~9.4s per (video, axis) ≈ ~13 min wall
- score_dir: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/`

### Result table

| Axis | baseline mean ± std | trained mean ± std | mean Δ | + wins | − losses | = ties | sign-test p (2-tail) |
|---|---|---|---|---|---|---|---|
| SA (prompt alignment) | 2.690 ± 0.715 | 2.667 ± 0.902 | −0.024 | 6 | 7 | 29 | 1.00 |
| PTV (temporal coherence) | 2.952 ± 0.731 | 2.762 ± 0.932 | **−0.190** | 4 | 13 | 25 | **0.049** ⚠️ |
| persistence | 3.357 ± 0.879 | 3.071 ± 0.973 | **−0.286** | 2 | 13 | 27 | **0.007** ⚠️ |
| inertia | 2.762 ± 0.692 | 2.714 ± 0.864 | −0.048 | 4 | 5 | 33 | 1.00 |
| momentum | 2.833 ± 0.730 | 2.714 ± 0.835 | −0.119 | 4 | 7 | 31 | 0.55 |

Sign-test 2-tailed exact binomial on (+wins, −losses) excluding ties.

### Critical regressions (multi-axis losers)
| prompt | regression pattern |
|--------|--------------------|
| `36e42af19937` | −2 across all 5 axes (catastrophic) |
| `e90b3f54bffb` | −1 on 4/5 axes |
| `2455740c4d45` | −1 across all 5 axes |
| `7977e8df650c` | −1 to −2 on 4/5 axes (PTV/persistence/inertia/momentum) |
| `48255a441729` | −1 on 4/5 axes |

### Few winners
| prompt | win pattern |
|--------|-------------|
| `242e01f46c08` | +1 across all 5 axes (best) |
| `e38a4396df92` | +1 on 4/5 axes |
| `70d3b1b89e19` | +2 on SA/PTV |

### Lesson #3: train metric ≠ video quality
- v3 acc_win50=0.88 + Q4 loss<0.1 occupancy 82% indicated successful pair-preference fitting.
- Same ckpt regressed downstream on PhyJudge persistence (−0.29, p=0.007) and PTV (−0.19, p=0.049).
- 5/42 prompts went catastrophically wrong at video level.
- **Round-N+ ckpts must pass downstream eval before being labeled "baseline" or "ratified".** train metrics are gates only on training stability, not method quality.

## Round-5 candidates (deferred for luke decision)

### Hypothesis A: training config (β=1000 + lora_rank=16) is the cause
- **Option A** stage-prepped pre-eval at recipes pin `e9380254b3e39385`: β=100 (10× softer sigmoid), lora_rank=8 (back to round-2 capacity, half the drift surface), lr=5e-5 keep, 200 step keep.
- Implies 1k tier_b is OK as data; the regime (β + rank combo) is the problem.

### Hypothesis B: tier_b pair noise leaks through preference signal
- 1k tier_b pairs come from `wmbench` human-eval DB; winner ≠ truly better is possible (rater disagreement, axis mismatch with PhyJudge).
- This cannot be diagnosed by comparing against round-2 M4: M4 changes too many axes at once (pair count, β, lr, rank, steps, and pipeline era), so it is not a valid discriminator.
- Data-side follow-up needs a controlled label audit / pair-quality audit on the round-4 1k subset, or a controlled re-train that changes only the data selection while keeping the round-5 training regime fixed.

### Hypothesis C: both contribute
- Plausible, but not separable from existing round-4 artifacts alone.
- Round-5 should start with a controlled training-config ablation (Option A) and keep a separate data-quality audit path for tier_b.

## Artifacts inventory

### v3 (round-4 r5) ckpts
- raw FSDP-wrapped: `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T023311Z/lora_final.safetensors` (153 500 376 B, lora_rank=16, sha256 `18a5fd0be28ba0c7fedf6d295d84beb4268270f1a1c5508dc1aa9e44cd6ac689`)
- intermediate: `lora_step{50,100,150}.safetensors` 同 dir
- run_manifest.json 同 dir
- stripped (DiffSynth-loadable): `juyi-finetune:~/m4_lora_gen_out/v3_round4/lora_final_stripped.safetensors` (153 440 328 B, 800 keys, sha256 `d97157a7c139dfbd3a2b7947687ec529fff2244c5ef9f4e1ec2118a30707f45a`)

### v3 PhyJudge eval
- 84 videos (42 baseline + 42 trained): `juyi-videorl:~/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z/heldout_regen/`
- mirror on juyi-finetune for scoring: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/`
- 84 per-video score JSON + results.jsonl + summary.json: `juyi-finetune:~/gen_out/v3_full_strip_20260429T162925Z/scores/20260429T193214Z/`

### #39 (round-4 lr=1e-5 control) ckpt
- raw: `juyi-finetune:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T042438Z/lora_final.safetensors`
- stripped: same dir, `lora_final_stripped.safetensors`
- partial videos (24/84 baseline before kill): `juyi-finetune:~/gen_out/39_full_strip_20260429T173042Z/`

### Round-2 M4 ckpt (only retained ckpt from rounds before round-4)
- DiffSynth-converted: `juyi-finetune:~/m4_ckpt/lora_final_diffsynth.safetensors` (76 MB, lora_rank=8 round-2 era)

### Configs & pins (recipes/, all on origin/main)
- `wan22_i2v_a14b__round2_v0.yaml` (data preprocessing, immutable; recipe_id `6bef6e104cdd3442`)
- `training_config_round4.yaml` (round-4 r5: β=1000 lora_rank=16; pin `b44cb5193ef8552b`)
- `training_config_round4_lr1e5.yaml` (#39 lr=1e-5 control; pin `d99847e544236392`)
- `training_config_smokeD.yaml` (D smoke pin `9f1aafc8ddc66884`, never run since FSDP step 0 verifications already cover Hammer 1)
- `training_config_optionA.yaml` (Option A staged pin `e9380254b3e39385`, β=100 lora_rank=8 lr=5e-5, never run)

## Lessons learned (round-4 distilled)

1. **High-β DPO U-shape**: expected unlearning mid-run trough; require quartile / rolling-window aggregate to avoid premature amber → red-light. Step-N snapshots are biased.
2. **train metric ≠ video quality**: pair-pref accuracy and loss-saturation occupancy are necessary but insufficient gates. Round-N+ ckpts require downstream eval (PhyJudge or equivalent) before "baseline" label.
3. **FSDP-wrapped LoRA save format**: trainer must strip `_fsdp_wrapped_module.` prefix segments at save time so DiffSynth `pipe.load_lora` finds matching keys. Without this, downstream silently produces baseline-equal output. Permanent fix landed at `2d44f08`.
4. **Multi-rank manifest race**: any per-rank concurrent write to a shared manifest path needs rank-0-as-leader pattern. Atomic-write tmp file collisions cause `OSError(EEXIST)` race crashes. Fixed in `heldout_regen.py` (rank 0 writes marker + canonical; ranks 1..N-1 poll + write per-rank manifest).
5. **β knob is a regime, not a magnitude**: bumping β to compensate for "no movement" treats the wrong cause. Round-4 r4 stagnation at `ln(2)` was lr × step short, not β too small. β controls sigmoid steepness in the DPO loss; only matters when policy can generate margin (which requires sufficient lr + steps).
6. **Cross-box ssh + GitHub key topology**: `/shared/user60/worldmodel/` is FS-mounted on home boxes (nnmc59 / nnmc60) but **not** on remote GPU boxes (juyi-finetune / juyi-videorl). For shared-FS code → remote box: scp / rsync. For nnmc60 agents (`enfunan` GitHub identity, no LukeLIN-web collab): SSH-tunnel through juyi-* boxes which carry LukeLIN-web key (`~/.ssh/id_ed25519_juyi`); or scp the LukeLIN-web key locally. Workflow: commit on shared FS → ssh authorized box → `git push origin main`. Authorized boxes: nnmc59 (LukeLIN-web SSH key on nnmc59 user1), juyi-finetune, juyi-videorl. Recently extended to nnmc60 agents via key scp to home dir + ssh config alias.

## Status

- Round-4 closed without ratifiable ckpt.
- Pipeline (trainer + orchestrator + scorer) is method-validated end-to-end via the strip-fix smoke and the 84-video full eval.
- Round-5 entry awaits luke decision on Option A vs a controlled data-quality audit path; old M4 is not a discriminator.

## Co-discovery credits

- v3 training launch + monitor + final summary fetch: rl1
- #39 control train (lr=1e-5) + train metrics interp: rl7 + rl1
- Eval generator + cache_pipe + race fix + FSDP-prefix strip workaround: rl7
- Trainer-side FSDP-prefix strip fix (round-N+ no-manual-strip): rl9 task #42 commit `2d44f08`
- Scorer optimizations + per-axis paired-delta + sign-test analysis + bug detection (byte-equal trained=baseline): rl8 task #40
- Dispatcher coordination + retracted Option A push (then re-endorsed when downstream data flipped the read): rl2
