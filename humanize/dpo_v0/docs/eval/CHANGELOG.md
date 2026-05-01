# Eval changelog

This file records eval-set transitions (v1 → v2 etc). Each entry lists:
- trigger
- exact diff
- affected reference numbers (rendered "historical" not "canonical")
- inputs and outputs of the transition

## eval-v2 (effective 2026-05-01)

**Trigger**: luke1 directive `eb4da377` after round-5 lora_final n=42 verdict
on eval-v1 returned `fail` (canonical winner reverts to round-4 β=100
lora_final). Manifest pre-registered as `humanize/dpo_v0/docs/eval/eval_v2_changeset.json`.

**Rationale** (per manifest): rebalance class A:E from 12:3 to 15:1 to reduce
n=3 noise on E and add multi-body collision diversity for the round-6+ collision
class signal (Newton's cradle is class A's only "precise momentum conservation"
strong-constraint prompt in v1; v2 adds 3 more strong-constraint A prompts).

### Diff (PROMPT_CLASS.json, evalprompt.md)

| field | v1 | v2 | Δ |
|---|---|---|---|
| A 多体碰撞 | 12 | **15** | **+3** (`366d2a1252b3` rolling→multi-body collision; `5b7bb71f101d` 3-ball collision; `d8b29a78eed7` block-vs-block-gate) |
| B 破坏 | 9 | 9 | 0 |
| C 流体 | 6 | 6 | 0 |
| D 阴影 | 5 | 5 | 0 |
| E 链式 | 3 | **1** | **−2** (removed: `36e42af19937` 双排多米诺 — physics resembles A; `9d500eec2188` 手指戳CD — weakest causal chain) |
| F 滚动 | 4 | 4 | 0 |
| G 抛掷 | 3 | 3 | 0 |
| **total** | **42** | **43** | **+1** ⚠ |

> ⚠ **Manifest math discrepancy**: `eval_v2_changeset.json:target_distribution_v2.total`
> claims `42`, but the per-class fields sum to `43` (15+9+6+5+1+4+3=43).
> `downstream_actions_after_verdict[5]` says "other 39 reuse from prior eval-v1
> runs" implying 3 v1 drops, but `remove` field only lists 2 (both E).
> Resolution: trust the explicit `remove` (2 E pids only) + `add` (3 A pids)
> diff. Total = 43. The "total: 42" and "39 reuse" are typos in the manifest.
> Future v3 should fix the count fields if luke1 wants 42 total (would require
> dropping 1 more — which is not specified).

### Affected reference numbers (now historical)

- **round-4 β=100 lora_final n=42 axes-avg Δ = +0.114** ([ref:r4_lora_final_n42])
  is on **eval-v1 prompt set** and is no longer canonical for round-6+ winner
  comparisons. Use only for historical / process audit.
- **round-5 lora_final n=42 verdict = fail** (`ad8fd28`) was computed on
  eval-v1 paired sign-test; it is binding for the round-5 ckpt itself
  (round-4 wins) and stays canonical as a verdict, but its underlying axes-avg
  numbers are eval-v1 reference and now historical.
- All `round4_v3_lr1e5_*.md` numeric tables in `experiment-results/` are eval-v1
  references; do not compare directly to eval-v2 numbers across prompt sets.
  Cross-set comparison must use re-evaluated scores on the aligned set.

### Eval-v2 onboarding artifacts

- `humanize/dpo_v0/docs/eval/eval_v2_changeset.json` (luke1 manifest, frozen
  pre-result `4b28c95`)
- `humanize/dpo_v0/script/eval/pick_eval_v2_A_prompts.py` (picker script,
  provenance: manual override of auto-shuffle picks per manifest)
- `humanize/dpo_v0/eval/PROMPT_CLASS.json` (v2: 43 entries; was v1: 42)
- `humanize/dpo_v0/docs/eval/evalprompt.md` (v2 class headers + A/E
  re-balance + eval-v1 frozen reference table preserved)
- `humanize/dpo_v0/docs/exp-plan/round5_eval_criterion.md` (round-5 lora_final
  n=42 paired-sign-test criterion, locked pre-result `7fc498b`, verdict =
  fail per `ad8fd28`)

### New A pids — provenance + cond images

| pid | group_id | cond image | subtopic |
|---|---|---|---|
| `366d2a1252b3` | `068bff11-2c8d-444e-80b9-39c396d517fb` | `~/cond_imgs/0197_perspective-center_trimmed-weight-protects-duck.jpg` (or fallback) | rolling → multi-body collision (kettlebell + duck via tube) |
| `5b7bb71f101d` | `119cd32a-4770-4d09-9030-bc7c2fa76bbe` | `~/cond_imgs/0029_perspective-center_trimmed-ball-train.jpg` | 3-ball collision (no-string Newton-cradle geometry) |
| `d8b29a78eed7` | `3d55ff7b-13eb-4b58-9562-bb54d6da3343` | `~/cond_imgs/0173_perspective-center_trimmed-stable-blocks.jpg` | block-vs-block-gate rigid impact + structural collapse |

### Generation runs supporting v2 onboarding

- v3 baseline + round-4 β=100 lora_final on 3 new A pids:
  `juyi-finetune:~/gen_out/eval_v2_baseline_plus_r4_n3_20260501T031745Z/`
  - regen launched 03:17 UTC May 1 with `HELDOUT_REGEN_SKIP_COUNT_ASSERT=1` env
    var (heldout_regen.py hot-patch: 5 LOC env-var override of 579/42 hardcoded
    asserts; not yet committed)
  - 3 prompts on 4-rank, mode-batched (all baselines first, then all trained)
- round-5 lora_final on 3 new A pids:
  `juyi-finetune:~/gen_out/eval_v2_round5_lora_final_n3_<TS>/` with
  `--baseline-from` reuse of the above

### eval-v1 freeze marker

eval-v1 numbers are frozen at git tag `<tag-name-tbd>` (or commit `ad8fd28`
which is the round-5 lora_final verdict commit, just before eval-v2 takes
effect via `humanize/dpo_v0/eval/PROMPT_CLASS.json` v2 update). To reproduce
any eval-v1 read, check out at this commit and use the pre-`d177b1a`
PROMPT_CLASS.json (12 A / 3 E / 3 G).

### Hot-patches needing to be committed (non-blocking)

1. `humanize/dpo_v0/eval/heldout_regen.py` — `HELDOUT_REGEN_SKIP_COUNT_ASSERT=1`
   env-var override for 579/42 count asserts (5 LOC; on juyi-finetune local).
   Permanent fix: either commit the env-var path or land an explicit
   `--n-prompts-override` CLI flag.
2. `humanize/dpo_v0/eval/heldout_regen.py` — `--device cuda:${LOCAL_RANK}`
   per-rank binding (currently bash wrapper `/tmp/heldout_regen_rankdev_wrapper.sh`).
   Permanent fix: add `torch.cuda.set_device(local_rank)` in `main()`.
3. `humanize/dpo_v0/benchmark/physground_score.py` — gen_ts collision under
   multi-rank concurrent finish. Workaround: per-prompt `--run-dir` invocation.
   Permanent fix: include `prompt_id` in the run_id derivation.

These three are the open eval-infra debt. They should land as a single PR
after eval-v2 onboarding stabilizes.
