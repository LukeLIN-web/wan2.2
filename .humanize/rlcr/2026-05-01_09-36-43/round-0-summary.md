# Round 0 Summary — round-7 A 路 pipeline up

## What was implemented

**Local artifacts (all committed on `round7` @ `f79dd1f`):**
- `humanize/dpo_v0/recipes/training_config_round7_fresh_v2aligned_beta100.yaml` (β=100, lr=1e-5, lora_rank/alpha=16, num_samples=1024, sampling_band=[901,999], `subset_pair_ids_sha256_hex16: a262b7153f58c37f`).
- `humanize/dpo_v0/recipes/training_config_round7_fresh_v2aligned_beta100_sha256_pin` = `3431fd826fc5662a` (matches yaml sha256[:16]).
- `humanize/dpo_v0/script/sample/round7_v2aligned_1024.py` (forked from round-6's `round6_class_balanced.py`): quotas A:357 / B:214 / C:143 / D:119 / E:24 / F:95 / G:72 = 1024, oracle 43/43 v2 round-trip preserved verbatim, new `--b-mode {rescue,repeat}` flag (default `repeat` after decision-1 resolution).
- `humanize/dpo_v0/script/train/launch_round7_fresh_v2aligned_train_v3_fsdp_lr1e5_beta100.sh` (forked from round-6 launcher, no narrative header per `feedback_no_comments`): drops `--init-lora-from` (fresh from base), `--save-every 32`, `--save-optimizer-state true`, `--allow-repeated-pair-ids true`. Pin chain pre-baked.
- `humanize/dpo_v0/train/train_dpo_i2v.py` patched: new `--allow-repeated-pair-ids` flag (default false). When true, the subset-pair-ids filter switches from set-based ("intersection with manifest") to multiset-aware ("manifest must cover the unique subset; iterate the subset list as a multiset so duplicates emit additional training steps"). Default false preserves round-4/5/6 set-based filter and ordering verbatim.
- `humanize/dpo_v0/docs/exp-plan/round7draft.md` §"Decisions to lock at promote-time" #1 marked **RESOLVED (2026-05-01) as fallback (ii)** — see "Plan deviations" below.

**Remote pipeline state on juyi-videorl (branch `round7` @ `f79dd1f`):**
- Sampler ran: `humanize/dpo_v0/out/round7/20260501T180307Z/T3_round7_v2aligned_1024.json` produced. 1024 entries, 982 unique + 42 B-class deterministic repeats (salts `B-cond` / `B-rescue` / `B-repeat-pick`), per-class realised quotas exactly match plan table, oracle 43/43 ✅.
- E zero-buffer verify: 24/24 cond-present-E pids verified non-empty in the union of (round-4 + round-5 + round-6) latent manifests.
- Round-7 union latent manifest at `humanize/dpo_v0/latents/20260501T180502Z/tier_b_round7_v2aligned_1024/manifest.jsonl`: 1964 records (= 982 unique pids × 2 roles {winner, loser}); all latent files present on disk.
- Training **running** at PID 182050, log `humanize/dpo_v0/logs/launch_round7_fresh_v2aligned_train_v3_fsdp_lr1e5_beta100_20260501T180958Z.log`, run dir `humanize/dpo_v0/ckpts/20260501T181040Z/`. 8-rank FSDP, all 3 pins verified OK (recipe=6bef6e104cdd3442, train_cfg=3431fd826fc5662a, pair_ids=a262b7153f58c37f). Trainer log lines so far:
  - `[pair_ids pin] OK (multiset): a262b7153f58c37f (subset=1024, unique=982, duplicates=42, manifest_total=982)` ✅ multiset path validated end-to-end.
  - `[dataset] 1024 pairs (dropped 0 with missing cond images)` ✅ no cond-image drops.
  - `[cond-cache] full hit — skipping VAE load entirely` ✅ 161/161 unique cond images cached from earlier rounds.
  - `[prompt-cache] full hit — skipping T5 load entirely` ✅ 161/161 unique prompts cached.
  - `[schedule] control=sample num_samples=1024 max_steps=None steps_per_epoch=128 samples_per_step=8 target_steps=128 target_samples=1024` ✅ exactly 1 clean epoch (= 128 steps × 8 ranks × 1 sample).

**Process artifacts:**
- `.humanize/bitlesson.md` (new lesson `BL-20260501-union-manifest-pid-role` — see BitLesson Delta).
- `.humanize/rlcr/2026-05-01_09-36-43/{goal-tracker.md, round-0-contract.md, round-0-summary.md}`.

## Files created / modified

```
A  .humanize/bitlesson.md
A  .humanize/rlcr/2026-05-01_09-36-43/goal-tracker.md
A  .humanize/rlcr/2026-05-01_09-36-43/round-0-contract.md
A  humanize/dpo_v0/recipes/training_config_round7_fresh_v2aligned_beta100.yaml
A  humanize/dpo_v0/recipes/training_config_round7_fresh_v2aligned_beta100_sha256_pin
A  humanize/dpo_v0/script/sample/round7_v2aligned_1024.py            (761 lines)
A  humanize/dpo_v0/script/train/launch_round7_fresh_v2aligned_train_v3_fsdp_lr1e5_beta100.sh  (82 lines)
M  humanize/dpo_v0/train/train_dpo_i2v.py                            (+38 lines: flag + multiset branch)
M  humanize/dpo_v0/docs/exp-plan/round7draft.md                      (decision-1 resolved in-place)
```
Two commits on `round7`: `17d3c9f` (artifacts + decision-1 resolution) and `f79dd1f` (pin chain post-sampler-run). Both pushed to `origin/round7`.

## Tests / validation

- Trainer `argparse` accepts `--allow-repeated-pair-ids` (verified via `python3 train/train_dpo_i2v.py --help`).
- Sampler `argparse` accepts `--b-mode {rescue,repeat}` and validates "rescue mode requires --rescue-b-pair-ids-json" (verified via `--help`).
- Sampler module-load assertions: `sum(CANONICAL_QUOTAS.values()) == 1024 == CANONICAL_TOTAL`, dict matches plan table.
- Recipe yaml sha256 round-trip: `sha256(yaml)[:16] == sidecar pin == launcher.expect_train_cfg == 3431fd826fc5662a` ✅
- Sampler real run on juyi-videorl: oracle 43/43 v2 round-trip; per-class realised n exactly = `{A:357, B:214, C:143, D:119, E:24, F:95, G:72}`; total 1024 with 982 unique + 42 duplicates; pair_ids sha256[:16]=`a262b7153f58c37f`.
- E zero-buffer verify on juyi-videorl: 24/24 cond-present-E pids present in union latent manifest.
- Round-7 union latent manifest: 1964 records = 982 unique pids × 2 roles, all on-disk.
- Trainer first launch (T+0): pin chain accepted, FSDP init started — but failed at the new multiset assertion with `RuntimeError: latent manifest is missing 982 of 982 unique subset pair_ids`. Root cause: my first union-manifest pass deduped by `pair_id` alone, dropping one of the two role entries per pid. Fix: rebuild union with `(pair_id, role)` key. Manifest now has 1964 entries; second launch (PID 182050) passed `[pair_ids pin] OK (multiset)` and is currently in DiT FSDP init.

## Plan deviations

**Plan §"Decisions to lock at promote-time" item #1 — B rescue + E zero-buffer policy** was originally an open decision. In Round 0 we discovered that the 4 unique source switch-frame jpgs the 46 B-class disk-missing pids depend on are gone from `/shared/user60/worldmodel/rlvideo/videodpo/worldmodelbench/physics-IQ-benchmark/switch-frames/` and from any reachable mirror; rebuild was infeasible. luke1-equivalent authorisation given (user direction in Round 0 mid-flight) to take fallback **(ii)**: keep `num_samples=1024` but allow B-class repeated sampling. Implementation is purely additive on the trainer (new flag, default false → round-4/5/6 untouched) and a sampler mode flag (default `repeat` since rescue is dead for this dataset epoch). Plan doc updated in-place (DRAFT, pre-promote) to record the resolution.

**Codex routing override**: task 7 (pre-flight) was originally tagged `analyze` per RLCR routing rules (Codex executes), but Codex's sandbox lacked active `gcloud` credentials and could not SSH to juyi-videorl. User directed Claude to take over via Claude's own `gcloud` auth. Task retagged `coding+claude` and recorded as a deviation in goal-tracker Plan Evolution Log. Future RLCR loops should fix Codex sandbox auth so `analyze` routing remains honoured.

## Remaining items

- **AC6 (training completion)**: in flight. Expected emit cadence: ckpts at step-32 / 64 / 96 / lora_final under `ckpts/20260501T181040Z/`. Will not complete within Round 0 (1 epoch on 1024 / 8 ranks ≈ several GPU-hours). Background monitor `b2f4n1gox` is watching for first step + grad_norm + checkpoint events.
- **AC7 (per-ckpt eval on juyi-finetune)**: deferred to Round 1+. Eval orchestration (gen + score per ckpt against eval-v2 n=43 baseline + paired sign-test vs round-4 anchor) starts naturally when step-32 ckpt lands.
- **Plan promote**: `round7draft.md` → `round7_plan.md` rename + freeze commit. Gated on luke1 sign-off and pre-flight outcome (now resolved); deferred per goal-tracker. Must happen before any round-7 lora_final eval is aggregated (per `howtoreport.md` §一 #5).

## BitLesson Delta

Action: `add`
Lesson ID(s): `BL-20260501-union-manifest-pid-role`
Notes: round-7 Round 0 spent one training-launch attempt rediscovering that DPO latent manifests in this repo are keyed by `(pair_id, role)` pairs, not by `pair_id` alone. Trainer's `load_pair_records` filters out any pid whose `roles` set lacks both `winner` and `loser`. When building a union manifest from multiple rounds' jsonls, dedup must use `(pair_id, role)` as the key — naive `records_by_pid[pid] = rec` drops one role and triggers `RuntimeError: latent manifest is missing N of N unique subset pair_ids`. New BitLesson entry added to `.humanize/bitlesson.md` so future round-N+1 union-manifest builders fix this on the first launch.
