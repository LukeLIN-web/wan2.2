# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Bring up the round-7 A 路 pipeline (fresh-init from Wan2.2-I2V-A14B base on a v2-aligned 1024-pair budget, 1 clean epoch β=100 lr=1e-5) end-to-end: build local artifacts here, sample/train on juyi-videorl, eval each saved ckpt (step-32/64/96 rolling-read + lora_final verdict) on juyi-finetune against eval-v2 n=43 with the round-4 β=100 lora_final anchor.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->

- AC1 — **Local artifact set committed on `round7` branch**: forked recipe yaml `recipes/training_config_round7_fresh_v2aligned_beta100.yaml`, forked sampler `script/sample/round7_v2aligned_1024.py`, forked train launcher `script/train/launch_round7_fresh_v2aligned_train_v3_fsdp_lr1e5_beta100.sh`, plus recipe sha256-pin sidecar. No narrative header / change-log comments in `.sh` or `.yaml` (per `feedback_no_comments`).
- AC2 — **Sampler quotas + B-rescue logic match plan**: A:357 / B:214 (= 172 cond-present + 42 rescued) / C:143 / D:119 / E:24 / F:95 / G:72 = 1024; v2 classifier oracle round-trip 43/43 enforced before any sampling step (`return 3` on disagreement); rescued B pids are required to be in the disk-missing set, not in cond-present.
- AC3 — **Recipe yaml deltas vs round-6 v2aligned are the documented set only**: `num_samples: 1024`, `max_pairs: 1024`, `round_tag: round-7-fresh-v2aligned`, `seed_namespace: round7-fresh-tier_b-1024-cond-present`, `subset_pair_ids_sha256_hex16: <pin from sampler run>`. β / lr / lora_rank / lora_alpha / dpo_loss_kind / sampling_band / micro_batch unchanged.
- AC4 — **Train launcher reflects fresh-init**: NO `--init-lora-from`, `--save-every 32`, `--save-optimizer-state true`, recipe / pair-ids pins sourced from sidecar files. `target_steps` semantics: trainer derives `ceil(1024 / 8) = 128` (1 epoch, no wrap) from `max_pairs / world_size`.
- AC5 — **Remote sampler + manifest produced on juyi-videorl** (Codex-driven): `out/round7/<UTC>/T3_round7_v2aligned_1024.json` with realized per-class n exactly matching plan table, `pair_ids_round7_v2aligned_1024_sha256_hex16_pin` written. Pre-flight pre-condition: 42 usable rescued B pids + 24 usable E pids verified non-empty on disk before sampling.
- AC6 — **Training launched on juyi-videorl** (Codex-driven): torchrun on 8 ranks, ckpts emitted at step-32 / 64 / 96 / lora_final under `ckpts/<UTC>/`. `lora_final_optim.pt` saved (round-8 warm path). No halt / no OOM / no low-noise routing.
- AC7 — **Per-ckpt eval on juyi-finetune** (Codex-driven): n=43 v2 PhyJudge gen+score for step-32 / 64 / 96 (rolling-read-only at n=24 per `howtoreport.md` §一 #1) and lora_final at n=43; `baseline_sha256_match: true` asserted; verdict-binding test = paired sign-test (round-7 lora_final vs round-4 β=100 lora_final), `α=0.05`, two-sided, direction favoring round-7.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan import from `docs/exp-plan/round7draft.md` (status DRAFT). | First round of RLCR loop. Plan promotion to `round7_plan.md` is luke1-gated and blocked on B-rescue feasibility (decision 1) — not done in Round 0. | n/a |
| 0 | Pre-flight (task 7) returned `PREFLIGHT_HALT` from Codex; downstream Codex tasks 8/9/10/11 NOT dispatched. | Codex sandbox could not run `gcloud compute ssh juyi-videorl` (no active gcloud account) and did not find the round-5/6 union latent manifest under `/shared/user60/...` — only the round-4 manifest at `latents/20260428T164115Z/tier_b_round4_1k/manifest.jsonl` is mirrored. So 46 raw B disk-missing pids were correctly classified (confirms plan), but cond-image rebuild + latent rebuild + 24 E latent verification could not be executed remotely. | AC5/AC6/AC7 blocked until env unblocked (or rescue manifest produced manually by operator). AC1/AC4 unaffected (local artifacts complete). |
| 0 | Task 7 retagged `analyze→coding+claude` after the env block. From my own shell `gcloud compute ssh juyi-videorl --zone=us-east5-a` works (auth account `luke.juyi.lin@gmail.com`); confirmed 46 B disk-missing classification independently against `/shared/...vidoepoWan/humanize/dpo_v0/out/round4/20260428T160839Z/drop_log.json`; the 46 B pids reference only **4 unique source jpgs** and all 4 jpgs are gone from disk anywhere reachable (`videodpo/worldmodelbench/physics-IQ-benchmark/switch-frames/` directory does not exist; same with `videodpoWan/WorldModelBench/images/`). | This confirms the rescue path is permanently infeasible for this dataset epoch. |
| 0 | Decision 1 of plan `round7draft.md` §"Decisions to lock at promote-time" resolved as fallback **(ii)** by user direction. | Source data unrecoverable; (i) drop-to-824 trades 200 pairs of training budget; (ii) keeps 1024 with 42 B-class repeats and matches the plan's existing target. User authorised (ii). | Adds two artifacts: `train/train_dpo_i2v.py` patch (new `--allow-repeated-pair-ids` flag, multiset-aware subset filter, default false); sampler `--b-mode {rescue,repeat}` flag (default `repeat`). Launcher passes `--allow-repeated-pair-ids true`. Plan §"Decisions to lock at promote-time" #1 updated in-place to record resolution. AC1/AC2/AC3/AC4 still met; AC5 narrows to "E zero-buffer + sampler+trainer integration" — B-rescue manifest no longer needed. |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| Initialize Goal Tracker (ACs from plan) | n/a (meta) | in_progress | coding | claude | Round 0 setup. |
| Write Round 0 contract | n/a (meta) | pending | coding | claude | |
| Fork recipe yaml | AC1, AC3 | pending | coding | claude | `subset_pair_ids_sha256_hex16` placeholder, filled by Codex post-sampler. |
| Fork sampler script | AC1, AC2 | pending | coding | claude | New quotas + B-rescue input, oracle 43/43 retained. |
| Fork train launcher | AC1, AC4 | pending | coding | claude | Drop `--init-lora-from`; `--save-every 32`. |
| Compute recipe sha256 pin sidecar | AC1, AC3 | pending | coding | claude | First-pass pin (will be updated when sampler-produced subset hash lands). |
| B-rescue + E zero-buffer pre-flight on juyi-videorl | AC5 | pending | analyze | codex | Decision 1 — halt if cannot produce 42 usable rescued B pids. |
| Build round-7 1024-pair manifest on juyi-videorl | AC5 | pending | analyze | codex | After pre-flight passes. |
| Build union latent manifest for round-7 | AC5, AC6 | pending | analyze | codex | Concat r4+r5+r6+rescued-B latents → 1024 manifest. |
| Launch round-7 training on juyi-videorl | AC6 | pending | analyze | codex | torchrun 8-rank FSDP, save_every=32, save-optimizer-state on. |
| Set up per-ckpt eval orchestration on juyi-finetune | AC7 | pending | analyze | codex | Reuse round-6 v2 baseline scores; rolling-read n=24 for step-32/64/96, n=43 verdict for lora_final. |
| Commit round-0 artifacts + write round-0-summary.md | AC1 | pending | coding | claude | |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| Plan-status DRAFT: not yet promoted to `round7_plan.md`; promotion is luke1-gated and conditional on B-rescue decision (decision 1). | 0 | AC5, AC6, AC7 (gates ckpt-eval aggregation per `howtoreport.md` §一 #5) | Pre-flight pre-condition outcome (Codex task) → luke1 sign-off → rename DRAFT → freeze commit at HEAD. Do NOT aggregate any round-7 ckpt eval before promote. |
| Codex sandbox cannot SSH to juyi-videorl (no active `gcloud` account; direct ssh blocked). | 0 | downstream `analyze`-tagged tasks (8/9/10/11) | Workaround applied: tasks that require remote execution are run from Claude's own shell (which has `gcloud` auth). Future RLCR loops should fix the Codex sandbox auth so `analyze` routing remains honoured. |

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| `subset_pair_ids_sha256_hex16` cannot be filled before the Codex sampler run on juyi-videorl. | 0 | Recipe yaml is correct in structure; the trainer's pair-ids pin is checked against a sidecar that gets updated post-sampling, so artifact creation is not blocked. | After Codex finishes "Build round-7 1024-pair manifest" — re-pin yaml + sha256 sidecar in a follow-up commit. |
| Eval-v2 r4 lora_final n=43 reference (decision 4) reuses round-6's merged jsonl; no new merge needed. | 0 | Already specified in plan as reuse-as-is (`round4_beta100_lora_final_v2_n43_flat.jsonl`). | n/a — informational. |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
| Promote `round7draft.md` → `round7_plan.md` (rename + freeze commit) | n/a (process) | 0 | Plan §"Authoring + change control" gates promotion on (a) luke1 sign-off and (b) B-rescue feasibility decision. Both are upstream of any round-7 ckpt eval; promote happens at promote-time, not Round 0. | After AC5 pre-flight outcome + luke1 directive. |
