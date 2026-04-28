# wan2.2-i2v-A14B v0 i2v-orig-init DPO — PhyJudge-9B Paired Delta

> **Status:** placeholder. Round-1 commit lays the file down so the M6 →
> M8 hand-off has a stable destination; populated by
> `humanize/dpo_v0/judge_paired_delta.py` once M6 lands the heldout
> videos and the harness is run for both legs. The renderer overwrites
> this file end-to-end on each run (see `write_results_md`); do not
> hand-edit until the run lands and you are amending notes around it.

> Comparator pair: **original-init no-DPO baseline vs original-init DPO trained**.
> Recipe id `6bef6e104cdd3442` (frozen). Aggregation rule `cross_group_rater_union` (frozen).

## Comparator

| Leg | Checkpoint path |
|-----|-----------------|
| baseline (original-init no-DPO) | _filled at run time_ |
| trained (original-init DPO) | _filled at run time_ |

## Provenance

| Field | Value |
|-------|-------|
| code commit id | _filled at run time_ |
| machine internal-IP tail | _filled at run time_ |
| compute envelope | _filled at run time_ |
| generation_config sha256[:16] | _filled at run time_ |
| pairs used | _filled at run time_ |
| baseline-only ids (excluded) | _filled at run time_ |
| trained-only ids (excluded) | _filled at run time_ |
| produced at (UTC) | _filled at run time_ |

## Judge probe

> Probed field-name list (top-level ∪ per-result, sorted). Exact bytes
> stamped here so reviewers can confirm probe-time field names match
> the operating contract.

```
_filled at run time_
```

Axis mapping (semantic → probed key):

| Semantic axis | Probed key |
|---------------|-----------|
| SA | _filled at run time_ |
| PTV | _filled at run time_ |
| persistence | _filled at run time_ |

## Primary composite delta — `SA + PTV + persistence`

| mean | std | 95% CI low | 95% CI high |
|------|-----|------------|-------------|
| _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Secondary axes

> One row per secondary axis returned by the probe. The 95% CI uses the
> same paired-bootstrap pair-index resamples as the composite (per-axis
> seed derived from the top-level seed) so the breakdown stays
> internally consistent with the primary scalar.

_filled at run time_

## Failure record

`null (no halt)` once the run completes; `{"kind": "judge-axis-missing", …}`
or similar if the run halts on the hard-contract check.

---

## Round-1 hand-off contract

The round-1 commit puts:

* `humanize/dpo_v0/judge_paired_delta.py` — probe + composite + paired
  bootstrap + manifest stamping + md writer in one module.
* `humanize/dpo_v0/test_judge_paired_delta.py` — pytest unit tests for
  every halt condition and the happy path.
* this template — destination for the renderer.

The wiring that still needs M6 / M5 outputs (real heldout videos and
their judge-evaluated JSONs) lands in a follow-up commit, not in
round-1.

PhyJudge harness invocation (`run_full_judge_eval`) is implemented but
gated behind `--actually-run-judge`; the round-1 surface is exercised
purely with pre-existing eval JSONs, which keeps the math and the
halt-on-missing contract verifiable in unit tests today.
