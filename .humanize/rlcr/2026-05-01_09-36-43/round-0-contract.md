# Round 0 Contract

## Mainline Objective

Bring the round-7 A 路 pipeline up: ship the local artifact set (recipe / sampler / launcher + sha256 pin sidecar) on the `round7` branch, then hand the remote pipeline (pre-flight → sample → train → eval) to Codex via `analyze`-tagged tasks for execution on juyi-videorl + juyi-finetune.

## Target ACs

- **AC1** — Local artifact set committed on `round7`.
- **AC4** — Train launcher reflects fresh-init.

(AC2, AC3 are checked structurally during artifact authoring; AC5/AC6/AC7 cover remote-execution outcomes that complete in later rounds and live with the Codex-tagged tasks.)

## Blocking Side Issues In Scope This Round

None. Plan-status DRAFT is tracked in goal-tracker and does not block local artifact creation; it only blocks _aggregation_ of any round-7 eval result (a downstream-round concern).

## Queued Side Issues Out Of Scope This Round

- Filling `subset_pair_ids_sha256_hex16` in the recipe yaml (depends on Codex sampler run on juyi-videorl) — re-pin in a follow-up commit.
- Promoting `round7draft.md` → `round7_plan.md` — luke1-gated, deferred per goal-tracker.

## Round Success Criteria

1. `recipes/training_config_round7_fresh_v2aligned_beta100.yaml` exists with the 5 documented deltas vs round-6 v2aligned; β / lr / lora_rank / lora_alpha / dpo_loss_kind / sampling_band / micro_batch unchanged.
2. `recipes/training_config_round7_fresh_v2aligned_beta100_sha256_pin` sidecar exists, value matches `sha256(yaml)[:16]`.
3. `script/sample/round7_v2aligned_1024.py` exists; `CANONICAL_QUOTAS` totals 1024 with the plan's per-class values; oracle 43/43 assertion is preserved verbatim from the round-6 sampler; B-rescue inputs validated against the disk-missing set.
4. `script/train/launch_round7_fresh_v2aligned_train_v3_fsdp_lr1e5_beta100.sh` exists with no `--init-lora-from`, `--save-every 32`, `--save-optimizer-state true`; no narrative header comments.
5. All five files are tracked + committed on the `round7` branch.
6. Codex-tagged remote tasks remain `pending` in goal-tracker (Round 0 does not aggregate remote outcomes).
