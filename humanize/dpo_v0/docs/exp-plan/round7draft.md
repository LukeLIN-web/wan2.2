# Round-7 plan (A 路) — fresh-init from base + v2-aligned 1024-pair (DRAFT)

**Status**: `DRAFT`. Round-6 lora_final verdict is now `fail`; this draft is
eligible to promote to `round7_plan.md` after the open decisions below are
resolved and the implementation artifacts are pinned.

- freeze before any round-7 ckpt eval is aggregated (per `howtoreport.md` §一 #5)

A 路 = "fresh-init from base + v2-aligned 1024-pair, 1 clean epoch". It tests
whether the round-4 warm chain is saturated while also increasing the effective
training budget beyond round-6. This is **not** a pure init-only ablation; it
changes both init and data volume/rescue policy.

## Background

- Round-5 warm (β=100 + 1202 setminus pairs, same A-heavy distribution) failed
  paired sign-test vs round-4 lora_final, p=0.0026 favoring round-4
  (`docs/experiment-results/round5_v3_lr1e5_warm_summary.md`).
- Round-6 warm (β=100 + 800 v2-aligned pairs, warm-from-r4) failed the
  verdict-binding eval-v2 n=43 paired sign-test vs round-4 lora_final:
  n_pos=5 / n_neg=19 / n_zero=19, p=0.0066 favoring round-4
  (`docs/experiment-results/round6_v2aligned_lora_final.md`).
- A 路 keeps the v2 eval-aligned target distribution and training hyperparams,
  but changes two things vs round-6: **fresh init from base** and **larger
  effective data** (1024 vs 800 via B-class rescue). If it beats round-4, it
  supports "fresh + larger aligned set" rather than isolating init alone.

## Hypothesis (single)

If round-7 trains on a **v2-aligned 1024-pair budget** initialised **fresh from
Wan2.2-I2V-A14B base**, then it beats round-4 β=100 lora_final on the v2 n=43
paired sign-test. The 1024 set is no-repeat after rescuing 42 B-class
disk-missing pairs into the latent/cond-image pool; without that rescue, the
strict v2-aligned no-repeat cap is only 824 because cond-present B has 172
pairs. Training hyperparams are otherwise held fixed at the round-6 v2aligned
recipe (β=100, lr=1e-5, lora_rank=16, micro_batch=1, sampling_band=[901,999],
dpo_loss_kind=sigmoid, world_size=8).

## Training data sampling spec

### v2-aligned 1024-pair budget

Eval-v2 class fractions (from `eval/PROMPT_CLASS.json` v2 / `eval_v2_changeset.json`,
n=43): A 15 / B 9 / C 6 / D 5 / E 1 / F 4 / G 3.

Budget = 1024. Per-class quotas are computed by floor + largest-remainder to
sum to exactly 1024. This budget requires B-class rescue: round-6's actual
cond-present classified pool is A:1163/B:172/C:378/D:134/E:24/F:177/G:113
(unclassified:41), so strict v2-aligned no-repeat without rescue tops out at
824. Round-7 selects all 172 cond-present B pairs plus 42 of the 46
disk-missing B pairs after their cond images/latents are rebuilt.

| class | eval v2 (n=43) | round-7 target (1024 budget) | source pool | buffer | 操作 |
|---|---|---|---|---|---|
| A 多体碰撞 | 15 (34.9%) | **357** | cond-present 1163 | 806 | downsample 0.307× |
| **B 破坏/形变** | 9 (20.9%) | **214** | cond-present 172 + rescued disk-missing 42/46 | 4 rescue-buffer | use all cond B + deterministic 42 rescued B |
| C 流体 | 6 (14.0%) | **143** | 378 | 235 | downsample 0.378× |
| D 阴影/反射 | 5 (11.6%) | **119** | 134 | 15 | use ~89% of pool |
| **E 链式** | 1 (2.3%) | **24** | cond-present 24 | **0 (zero-buffer)** | use 100% of pool |
| F 滚动 | 4 (9.3%) | **95** | 177 | 82 | downsample 0.537× |
| G 抛掷 | 3 (7.0%) | **72** | 113 | 41 | downsample 0.637× |
| **总计** | **43** | **1024** | **2161 cond-classified + 42 rescued B** | | |

### B rescue + E zero-buffer policy (decision required pre-freeze)

A 路 extends round-6's class-balanced sampler by adding a B-class rescue step.
E remains zero-buffer because all 24 cond-present E pairs are used. Risk: any
corrupted E pair, or more than 4 unusable rescued B candidates, invalidates the
1024 no-repeat plan. Mitigations:

- pre-flight check: rebuild/verify cond images + latents for the 46 raw
  disk-missing B candidates, then deterministically select 42; assert all
  selected rescued B pids and all 24 E pids have non-empty latent entries on
  juyi-videorl before training.
- if pre-flight cannot produce 42 usable rescued B pids, halt; either (i) drop
  to 824 strict cond-present no-repeat, or (ii) explicitly approve repeated
  B sampling with a trainer patch. Decision deferred to promote-time.

### Sampler determinism

- Source: `humanize/dpo_v0/out/round4/20260428T160839Z/` cond-present 2202
  manifest plus 42 rescued B-class pids from the raw tier_b disk-missing set
  (sha256-pinned via `T3_round4_tier_b_1k.json` + drop-log provenance chain).
- Per-class oracle: `docs/data/distirbution.md` rule, re-applied to v2's
  `eval/PROMPT_CLASS.json` (43 pids, 43/43 verify). Round-6 plan §"decision 2"
  carries forward.
- Sampler script (TBD): `humanize/dpo_v0/script/sample/round7_v2aligned_1024.py`
  (fork from round-6's `round6_class_balanced.py`; differs in quotas, seed
  namespace, and B-class rescue input).
- Output manifest: `T3_round7_v2aligned_1024.json` + `subset_pair_ids_sha256_hex16`
  pin, asserts per-class realized n equals the table above.
- Seed namespace: `round7-fresh-tier_b-1024-cond-present`.

## Training config spec

| field | value | delta vs round-6 v2aligned |
|---|---|---|
| init LoRA | **none (fresh from base)** | **CHANGED**: round-6 was warm from `ckpts/20260429T234925Z/lora_final.safetensors` |
| init optim state | none (cold AdamW) | unchanged |
| `--save-optimizer-state` | on | unchanged (enables round-8 warm-from-r7 path) |
| β | 100 | unchanged |
| lr | 1.0e-5 | unchanged |
| lora_rank / lora_alpha | 16 / 16 | unchanged |
| dpo_loss_kind | sigmoid | unchanged |
| sampling_band | [901, 999] | unchanged |
| micro_batch | 1 | unchanged |
| num_samples | **1024** | **CHANGED**: round-6 was 800 |
| target_steps | **128** (= ceil(1024/8), exactly 1 epoch, no wrap) | **CHANGED**: round-6 was 100 |
| save_every | **32** (4 救生圈 + lora_final at step-32/64/96/128) | **CHANGED**: round-6 was 20 |
| seed_namespace | **`round7-fresh-tier_b-1024-cond-present`** | renamed |
| round_tag | **`round-7-fresh-v2aligned`** | renamed |
| recipe yaml | **`recipes/training_config_round7_fresh_v2aligned_beta100.yaml`** | new fork from `training_config_round6_v2aligned_beta100.yaml` |

Hard contracts (carried forward from round-4/5/6): total_forwards == num_samples
× (target_steps × micro_batch × world_size / num_samples) per F.11; routing
100% high-noise; no halt / no OOM; baseline_sha256_match: true at eval time.

### AdamW cold-start expectation

round-7 is cold-AdamW (round-4 didn't save optim state, so even round-6 was
already cold). Expect grad_norm spike + margin volatility in steps 0–10 as
momentum buffers warmup. **Not** an amber halt signal. round-7's
`--save-optimizer-state on` means round-8 (whatever its plan) can warm-resume
optim if desired.

## Eval criterion (pre-registered, to be locked at promote-time)

> **Round-7 lora_final n=43 verdict = `pass`** iff paired sign-test
> (round-7 lora_final, round-4 β=100 lora_final) on **identical 43 v2
> prompt-ids**, per-prompt-axes-avg-Δ sign vs 0 (two-sided binomial,
> α=0.05), with the test direction set to **favoring round-7**.
>
> Else `verdict = fail` (canonical winner stays round-4 lora_final;
> round-7 ckpt is not adopted).

Anchor = round-4 β=100 lora_final (same as round-6 criterion). Not vs round-6
— that comparison is supplementary (see below) and not verdict-binding.

### Operational definitions

Identical to round-6 plan §"Eval criterion / Operational definitions". Reused
v2 references:

| field | source |
|---|---|
| round-7 lora_final scores | `juyi-finetune:~/gen_out/round7_lora_final_n43_<TS>/scores_perprompt/<pid>/<run_ts>/results.jsonl` |
| round-4 β=100 lora_final v2 scores (40 reused + 3 new A pids) | `round4_beta100_lora_final_v2_n43_flat.jsonl` (built per round-6 plan §"decision 4"; reused as-is) |
| baseline videos (v2) | reuse round-6's v2 baseline assembly |
| baseline scores (v2) | reuse round-6's v2 baseline scores |

`baseline_sha256_match: true` asserted at runtime; failure voids the round-7
result md per `howtoreport.md` §一 #10.

### Supplementary tests (informational, not verdict-binding)

- **fresh+larger-vs-warm**: round-7 lora_final vs round-6 lora_final paired
  sign-test on the same n=43 v2 prompts. Answers "did the fresh-init +
  1024-pair rescue plan outperform round-6 warm + 800?" Reported in result md
  as a separate row, not as a verdict gate.
- **per-class supplementary**: 5×7 axes-class table with Bonferroni
  α_corrected = 0.05/35. Same format as round-6. Informational only.
- **early-ckpt rolling reads**: step-32/64/96 PhyJudge eval at n=24 for
  trajectory direction-of-travel only (`rolling-read-only`,
  `howtoreport.md` §一 #1). No verdict on non-final ckpts.

### Stats implementation

`humanize/dpo_v0/eval/stats.py` (rl8). Reused functions: `paired_sign_test`,
`bootstrap_ci`, `sign_test_vs_zero`. No hand-math in result md.

## Decision tree

| outcome | action |
|---|---|
| `pass` (p<0.05 favoring round-7) | adopt round-7 lora_final as canonical winner; commit `lora_final_optim.pt` for round-8 warm-resume; freeze v2 r7 reference numbers in CHANGELOG. If supplementary fresh+larger-vs-warm also p<0.05 favoring round-7 → evidence that leaving the round-4 warm chain plus increasing aligned data was the right move |
| `draw` (p ≥ 0.05) | round-4 stays canonical. Diagnose: (i) data plan still insufficient (B+D pool 回填 + bigger N); (ii) recipe knobs need revisit (e.g., longer training — 1 epoch may be too short; round-4 lr1e5 winner needed 250 steps = 2 epochs on 1k); (iii) base model has narrow physics movement budget regardless of init. Round-8 plan informed by which of (i/ii/iii) the supplementary signals point at |
| `fail` (p<0.05 favoring round-4) | round-4 stays canonical. Strong signal "v2-aligned 1024 + fresh + 1 epoch" undertrained or wrong recipe. Round-8 should explore (i) longer training (target_steps=200 or 250 on 1024-pair, accept 1.6–2 epoch with wrap), or (ii) lr bump (lr=2e-5 to compensate for short budget) |

In all three branches, round-7 `lora_final_optim.pt` is preserved so round-8
can choose between (i) warm-from-r7 with optim, (ii) warm-from-r4 with
re-trained optim (requires r4 re-run), (iii) fresh again with adjusted recipe.

## Decisions to lock at promote-time (open)

1. **B rescue + E zero-buffer policy** — **RESOLVED (Round 0, 2026-05-01) as
   fallback (ii)**: the 4 unique source switch-frame jpgs the 46 B-class
   disk-missing pids depend on are no longer present at
   `/shared/user60/worldmodel/rlvideo/videodpo/worldmodelbench/physics-IQ-benchmark/switch-frames/`
   nor any reachable mirror; rebuild was infeasible. luke1-equivalent
   authorisation was given to take fallback (ii): keep `num_samples=1024`
   but allow B-class repeated sampling. Implementation:
   `humanize/dpo_v0/script/sample/round7_v2aligned_1024.py` defaults to
   `--b-mode repeat` (172 cond-present-B + 42 deterministic repeats from
   the same pool, salts `B-cond` / `B-rescue` / `B-repeat-pick`); trainer
   `humanize/dpo_v0/train/train_dpo_i2v.py` adds `--allow-repeated-pair-ids`
   (default false; round-7 launcher sets it true). Round-4/5/6 default
   semantics are unchanged. E zero-buffer pre-flight (24/24 cond-present-E
   non-empty in the trainer union latent manifest) still pending verify on
   juyi-videorl pre-launch.
2. **v2 classifier re-verify**: rerun rule on v2 PROMPT_CLASS (43 pids), assert
   43/43; same gate as round-6 plan §"decision 2".
3. **Recipe yaml**: fork from `training_config_round6_v2aligned_beta100.yaml`
   with the deltas above. Pin sha256_hex16 of the new yaml. Per
   `feedback_no_comments.md`: no narrative header / change-log comments in the
   yaml itself.
4. **Eval-v2 r4 lora_final n=43 reference**: reuse the merged jsonl built per
   round-6 plan §"decision 4". No new merge required.
5. **target_steps**: pre-truncate `num_samples=1024` in the recipe; default
   formula `ceil(1024/8)=128` lands exactly on the cap. No `--max-steps-override`
   patch required. 128 × 8 = 1024 pairs/epoch = exactly 1 clean epoch.
6. **save_every cadence**: 32 (4 ckpts + lora_final). Alternative 16 (8 ckpts)
   considered but adds ~1.5 GB ckpt disk per save × 8 = ~12 GB; round-6's 20-step
   cadence at 100-step run produced 5 ckpts and was workable, so 32-step
   cadence at 128-step run keeps similar density. Default: 32.

## Source data pinning

- Round-6 verdict (the gate for promoting this draft):
  `docs/experiment-results/round6_v2aligned_lora_final.md` (verdict = `fail`,
  commit `f07b8cc`).
- Eval-v2 manifest: `docs/eval/eval_v2_changeset.json` (luke1 `4b28c95`).
- Eval-v2 changelog: `docs/eval/CHANGELOG.md`.
- Class distribution analysis: `docs/data/distirbution.md` (eval-v1; v2
  classifier re-verify per decision 2 above).
- Round-6 plan (forked structure): `docs/exp-plan/round6_plan.md`.
- Round-6 recipe (forked yaml): `recipes/training_config_round6_v2aligned_beta100.yaml`.
- Pool manifest source: `humanize/dpo_v0/out/round4/20260428T160839Z/T3_round4_tier_b_1k.json`
  (and the 2202 cond-present derivation chain) plus
  `humanize/dpo_v0/out/round4/20260428T160839Z/drop_log.json` for the
  B-class disk-missing rescue candidates.
- round-4 β=100 lora_final ckpt (NOT used as init, but as eval anchor):
  `juyi-videorl:~/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234925Z/lora_final.safetensors`.
- Wan2.2-I2V-A14B base (init source): per `recipes/wan22_i2v_a14b__round2_v0.yaml`
  data preprocessing recipe (recipe_id `6bef6e104cdd3442`); base model path
  same as round-2/3/4/5/6 trainer entrypoint.

## Authoring + change control

- **Author**: TBD (draft; pickup at promote-time)
- **Pre-registration freeze**: this doc is `DRAFT`. Promote = rename to
  `round7_plan.md` + commit at HEAD before any round-7 lora_final score is
  aggregated. Post-result modification = process violation per
  `howtoreport.md` §一 #5.
- **Change-control after freeze**: any modification requires `--allow-empty`
  commit recording (1) timestamp of result aggregation, (2) proposer,
  (3) approver, (4) reason.
- **Discard condition**: obsolete; round-6 verdict is `fail`.

## Sign-off

- TBD — pending luke1 directive on promote/freeze and B-rescue policy.
