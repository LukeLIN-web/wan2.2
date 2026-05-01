# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

## Lesson: union-manifest-pid-role
Lesson ID: BL-20260501-union-manifest-pid-role
Scope: humanize/dpo_v0/train/train_dpo_i2v.py (`load_pair_records`); union latent manifest builders for any round that concatenates earlier-round manifests (e.g. `latents/<UTC>/tier_b_round7_*/manifest.jsonl`).
Problem Description: trainer fails at startup with `RuntimeError: latent manifest is missing N of N unique subset pair_ids` even though every `pair_id` from the subset *does* appear in the union manifest, because each pid has TWO entries (`role: winner` + `role: loser`) and the union-builder deduped by `pair_id` alone, dropping one role.
Root Cause: `load_pair_records` collects entries into `pairs_by_id.setdefault(pid, {})[role] = entry` and then filters `{pid: roles for pid, roles in pairs_by_id.items() if "winner" in roles and "loser" in roles}`. A union builder that does `records_by_pid[pid] = rec` keeps only the last-seen role per pid, so 100% of pids fail the both-roles gate and the trainer's downstream "subset minus manifest_intersect" check reports "missing N of N".
Solution: dedup the union by the composite key `(pair_id, role)`, not by `pair_id` alone. Expected post-filter count is `2 × len(unique_subset_pair_ids)`. Verify before launch: `len([r for r in union if r["pair_id"] in subset]) == 2 * len(set(subset))` and per-pid `roles == {"winner","loser"}`.
Constraints: applies to any DPO latent manifest derived from `dataprocessing/manifest_writer.py` style emitters where `(pair_id, role)` is the natural composite key. Subset-pair-ids list in `T3_*.json` files contains pair_ids only — the role split is implicit.
Validation Evidence: round-7 Round 0 first training launch (PID 181282) crashed at `train_dpo_i2v.py:1148` with the missing-982-of-982 message; second launch (PID 182050) after rebuilding the manifest with `(pid,role)` dedup passed `[pair_ids pin] OK (multiset)` and proceeded into FSDP init.
Source Rounds: round-7 Round 0 (2026-05-01).
