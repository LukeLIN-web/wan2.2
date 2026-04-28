"""PhyJudge-9B paired-delta scoring for the v0 i2v-original-init DPO run.

Terminal scoring entry point. Composes four steps:

1. **Probe**. Invokes the PhyJudge-9B harness (``serveandeval_9b.sh``) on a
   tiny smoke set, reads back the result JSON, and records the exact
   field-name list it returned plus an explicit ``axis_mapping`` table
   from the three semantic axes (``SA``, ``PTV``, ``persistence``) to the
   probed field names. The composite primary scalar is ``SA + PTV +
   persistence`` — matching the human-eval DB aggregation rule used to
   construct the preference pairs. If any of the three axes is not
   reachable from the probe (no exact-name match and no operator-supplied
   alias), the run halts with a ``judge-axis-missing`` failure rather
   than silently substituting another field. Recipe contract: see
   ``recipes/wan22_i2v_a14b__round2_v0.yaml`` and ``recipes/recipe_id``;
   the recipe id ``6bef6e104cdd3442`` is hard-coded here as the frozen
   value the trainer asserts at startup.

2. **Joined-result load**. Reads the per-leg eval JSON (saved by the
   ``serveandeval_9b.sh`` → ``run_eval.py`` → ``evals.vlm_eval`` pipeline)
   and pairs records by their ``video`` identifier. The heldout regen
   runner is responsible for using a single canonical id format
   (``<prompt_id>__seed<seed>`` is the convention used by M6) so that
   the same identifier is present on both legs. A pair survives only if
   *both* legs produced a record with that id; missing-on-one-side ids
   are reported and excluded from the bootstrap.

3. **Composite + paired bootstrap**. Per pair, the composite score is
   the sum of the three primary axes; the paired delta is
   ``trained_composite - baseline_composite``. Bootstrap resampling is
   done at the *pair index* level (with replacement), never at the
   per-leg level, so the within-pair correlation between baseline and
   trained is preserved. The same pair-index resamples are reused to
   compute CIs for every secondary axis the probe returned, which keeps
   the secondary breakdown internally consistent with the primary scalar.

4. **Manifest stamp**. Writes a JSON manifest with ``recipe_id``,
   ``aggregation_rule = cross_group_rater_union``, the probe's exact
   field-name list, the explicit axis-mapping table,
   ``compute_envelope`` (``single_gpu`` for the judge step), the code
   commit-id, the machine internal-IP tail, the trained-ckpt and
   baseline-ckpt paths, the generation_config SHA, and the bootstrap
   parameters. The terminal numbers also land in
   ``docs/experiment-results/wan14B_i2v_orig_init.md``.

Skeleton stage. The harness invocation is gated behind
``--actually-run-judge`` (default off); when off, the script consumes
pre-existing eval JSONs supplied via ``--baseline-results-json`` and
``--trained-results-json``. This keeps the composite + bootstrap math
testable end-to-end before the M6 heldout-regen videos and the real
judge run land. When on, ``run_full_judge_eval`` calls the harness
script via ``subprocess`` once per leg (with the appropriate video
prefix and prompt config) and writes the eval JSON to a timestamped
output directory.

Hard contracts honoured here:

* Heldout videos are NEVER pre-encoded (this module only consumes
  generated videos at eval time; the encode runner enforces the
  T3-subset heldout-leak guard separately).
* ``aggregation_rule = cross_group_rater_union`` is stamped as a string
  literal, never recomputed or substituted.
* Bootstrap is paired by pair-index. Independent-leg bootstrap is a
  category error and is not implemented.
* Missing-axis halts with ``judge-axis-missing`` and writes the failure
  record to the manifest; no silent substitution.

CLI:

    python judge_paired_delta.py \\
        --baseline-results-json ./eval_baseline.json \\
        --trained-results-json  ./eval_trained.json \\
        --baseline-ckpt-path    /path/to/wan22_i2v_a14b/orig_no_dpo \\
        --trained-ckpt-path     /path/to/wan22_i2v_a14b/orig_dpo_lora \\
        --generation-config-json ./gen_config.json \\
        --out-manifest          ./out/judge_manifest.json \\
        --out-md                ./docs/experiment-results/wan14B_i2v_orig_init.md \\
        --bootstrap-iters       10000 \\
        --bootstrap-seed        0xdpo_judge \\
        --comparator-pair       'original-init no-DPO baseline vs original-init DPO trained'
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import pathlib
import socket
import subprocess
import sys

# --- frozen contract values -------------------------------------------------

# Frozen recipe id from ``recipes/recipe_id`` (file content =
# ``sha256(canonical_yaml.bytes).hex[:16]``). The trainer recomputes this at
# startup and asserts equality before any forward pass; any drift here
# invalidates the run. Do not edit this constant — change the recipe YAML
# instead and re-stamp ``recipes/recipe_id``.
KNOWN_GOOD_RECIPE_ID = "6bef6e104cdd3442"

# Three semantic axes that compose the human-eval DB primary scalar. Any
# remap must go through ``axis_mapping`` (operator-supplied) and must still
# resolve all three names in the probe; missing any one is a hard halt.
COMPOSITE_AXES: tuple[str, ...] = ("SA", "PTV", "persistence")

# Aggregation rule stamp. Frozen as a literal — never recomputed. Drift
# on this value would invalidate the comparator pair.
AGGREGATION_RULE = "cross_group_rater_union"

# Default bootstrap iteration count. 10000 is well past the regime where
# the 95% CI shifts more than the score-discretization granularity for
# 42-pair samples; expose ``--bootstrap-iters`` so reviewers can verify
# stability empirically by re-running at a higher value.
DEFAULT_BOOTSTRAP_ITERS = 10000

# Default deterministic seed for the bootstrap resampler. Hex-y string
# so the source distinguishes it from training seeds at a glance.
DEFAULT_BOOTSTRAP_SEED = "0xdpo_judge"

# Default harness script (resolves to the wmbench-side PhyJudge-9B
# wrapper). The trainer probes this script once at startup; ``run_eval``
# writes its output JSON under ``data/scores/ourckpt/`` per its own
# ``--save_path`` argument.
DEFAULT_HARNESS_SCRIPT = (
    "/shared/user60/worldmodel/wmbench/evals/script/serveandeval_9b.sh"
)


# --- exceptions -------------------------------------------------------------

class JudgeAxisMissingError(RuntimeError):
    """Raised when the probe cannot resolve one of the composite axes.

    The caller catches this, writes the failure record to the manifest,
    and halts the run.
    """


# --- data records -----------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AxisMapping:
    """Per-semantic-axis source key in the probed JSON.

    ``mapping[name]`` is the per-result key in the probe's ``results[]``
    objects whose value is the score for the named semantic axis. Default
    is the identity map (``SA → SA`` etc.) — the operator supplies a
    non-identity map only if the probe returned a different spelling.
    """

    mapping: dict[str, str]

    def axis_key(self, semantic_name: str) -> str:
        if semantic_name not in self.mapping:
            raise JudgeAxisMissingError(
                f"axis_mapping has no entry for {semantic_name!r}"
            )
        return self.mapping[semantic_name]

    def to_serializable(self) -> dict[str, str]:
        return dict(self.mapping)


@dataclasses.dataclass(frozen=True)
class JudgedRecord:
    """One judged video result, normalized for paired delta arithmetic.

    Constructed from the raw ``results[]`` entry of an eval JSON via
    :func:`load_eval_results`. The ``primary_axes`` dict carries the three
    composite axis scores keyed by their *semantic* name (``SA``,
    ``PTV``, ``persistence``); the original probe field-name lives in
    the manifest, not on every record. ``secondary_axes`` carries the
    flat per-axis scalars the probe returned beyond the composite three;
    the secondary breakdown's CIs are computed against this dict.
    """

    video_id: str
    primary_axes: dict[str, float]
    secondary_axes: dict[str, float]
    raw: dict

    @property
    def composite(self) -> float:
        for axis in COMPOSITE_AXES:
            if axis not in self.primary_axes:
                raise JudgeAxisMissingError(
                    f"record {self.video_id!r} missing primary axis {axis!r}"
                )
        return float(sum(self.primary_axes[axis] for axis in COMPOSITE_AXES))


@dataclasses.dataclass(frozen=True)
class PairedDelta:
    """One per-pair delta record (trained minus baseline)."""

    video_id: str
    composite_delta: float
    secondary_deltas: dict[str, float]


@dataclasses.dataclass(frozen=True)
class BootstrapCI:
    """Mean / std / 95% percentile CI for one scalar via paired bootstrap."""

    mean: float
    std: float
    ci_low: float
    ci_high: float

    def to_serializable(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


# --- probe ------------------------------------------------------------------

def probe_phyjudge_axes(
    probe_results_json: pathlib.Path,
    axis_mapping_override: dict[str, str] | None = None,
) -> tuple[list[str], AxisMapping]:
    """Read a single probe-eval JSON and resolve the axis mapping.

    The caller is responsible for invoking ``serveandeval_9b.sh`` with a
    smoke video (``--limit 1`` etc.) before calling this function; the
    probe step does not run the harness itself, it only reads back what
    the harness wrote. This split keeps the probe deterministic when the
    skeleton is exercised against pre-computed JSONs.

    Returns ``(field_names, axis_mapping)``:

    * ``field_names`` — the union of top-level keys and ``results[0]``
      keys, sorted, ASCII-clean. Stamped verbatim in the manifest.
    * ``axis_mapping`` — for each semantic axis (``SA``, ``PTV``,
      ``persistence``), the per-result key whose value is that axis's
      score. Defaults to identity. ``axis_mapping_override`` lets the
      operator supply a non-identity map (e.g. ``{"SA":
      "alignment_score"}``) which is then validated against the probe's
      keys; an override that names a non-existent key halts.

    Halts with :class:`JudgeAxisMissingError` if any composite axis is
    unreachable.
    """
    payload = json.loads(probe_results_json.read_text())
    if not isinstance(payload, dict):
        raise JudgeAxisMissingError(
            f"probe JSON must be an object, got {type(payload).__name__}"
        )
    results = payload.get("results")
    if not isinstance(results, list) or len(results) == 0:
        raise JudgeAxisMissingError(
            f"probe JSON has no results[] (got {type(results).__name__}, "
            f"len={len(results) if isinstance(results, list) else 'n/a'})"
        )

    first_result = results[0]
    if not isinstance(first_result, dict):
        raise JudgeAxisMissingError(
            f"probe results[0] must be an object, got {type(first_result).__name__}"
        )

    top_level = sorted(payload.keys())
    per_result = sorted(first_result.keys())
    field_names = sorted(set(top_level) | set(per_result))

    # Resolve axis mapping. Default is identity; override may rename.
    mapping: dict[str, str] = {}
    for axis in COMPOSITE_AXES:
        if axis_mapping_override and axis in axis_mapping_override:
            target = axis_mapping_override[axis]
        else:
            target = axis
        if target not in first_result:
            raise JudgeAxisMissingError(
                f"composite axis {axis!r} maps to {target!r} but probe "
                f"results[0] has no such key (available: {per_result})"
            )
        score = first_result[target]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise JudgeAxisMissingError(
                f"composite axis {axis!r} → {target!r} is not numeric "
                f"(got {type(score).__name__})"
            )
        mapping[axis] = target

    return field_names, AxisMapping(mapping=mapping)


def secondary_axis_keys(
    sample_record: dict,
    axis_mapping: AxisMapping,
) -> list[str]:
    """Return the sorted list of per-result keys eligible as secondary axes.

    A key is eligible if its value is a finite number AND it is not one
    of the three primary axis source keys. Non-numeric fields (strings,
    nested dicts, lists) are excluded. The composite ``general_avg`` is
    excluded because it is a deterministic function of the primary axes
    and would inflate the secondary count without adding information.
    """
    primary_keys = {axis_mapping.axis_key(a) for a in COMPOSITE_AXES}
    excluded = primary_keys | {"general_avg"}

    out = []
    for key, value in sample_record.items():
        if key in excluded:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(key)
    return sorted(out)


# --- result ingestion -------------------------------------------------------

def load_eval_results(
    path: pathlib.Path,
    axis_mapping: AxisMapping,
    secondary_axis_keys_list: list[str],
) -> list[JudgedRecord]:
    """Load one leg's eval JSON and normalize each result to a JudgedRecord.

    ``video_id`` is taken from the per-result ``video`` field; the M6
    heldout regen runner is responsible for using a consistent id (e.g.
    ``<prompt_id>__seed<seed>``) so the two legs join cleanly.

    Halts with :class:`JudgeAxisMissingError` if any record is missing
    one of the composite axis source keys, or if any secondary axis
    listed in ``secondary_axis_keys_list`` is missing for any record.
    Halt-on-missing is by design — silent substitution would corrupt
    the paired delta.
    """
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "results" not in payload:
        raise JudgeAxisMissingError(f"eval JSON {path} has no results[]")

    out: list[JudgedRecord] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(payload["results"]):
        if not isinstance(raw, dict):
            raise JudgeAxisMissingError(
                f"eval JSON {path} result[{idx}] is not an object"
            )
        video_id = raw.get("video")
        if not isinstance(video_id, str) or video_id == "":
            raise JudgeAxisMissingError(
                f"eval JSON {path} result[{idx}] has no 'video' string id"
            )
        if video_id in seen_ids:
            raise JudgeAxisMissingError(
                f"eval JSON {path} duplicate video id {video_id!r}"
            )
        seen_ids.add(video_id)

        primary: dict[str, float] = {}
        for axis in COMPOSITE_AXES:
            key = axis_mapping.axis_key(axis)
            if key not in raw:
                raise JudgeAxisMissingError(
                    f"eval JSON {path} result {video_id!r} missing "
                    f"primary key {key!r} (axis {axis!r})"
                )
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise JudgeAxisMissingError(
                    f"eval JSON {path} result {video_id!r} primary key "
                    f"{key!r} is not numeric (got {type(value).__name__})"
                )
            primary[axis] = float(value)

        secondary: dict[str, float] = {}
        for key in secondary_axis_keys_list:
            if key not in raw:
                raise JudgeAxisMissingError(
                    f"eval JSON {path} result {video_id!r} missing "
                    f"secondary axis {key!r}"
                )
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise JudgeAxisMissingError(
                    f"eval JSON {path} result {video_id!r} secondary axis "
                    f"{key!r} is not numeric (got {type(value).__name__})"
                )
            secondary[key] = float(value)

        out.append(JudgedRecord(
            video_id=video_id,
            primary_axes=primary,
            secondary_axes=secondary,
            raw=raw,
        ))

    return out


def pair_records_by_video_id(
    baseline: list[JudgedRecord],
    trained: list[JudgedRecord],
) -> tuple[list[PairedDelta], list[str], list[str]]:
    """Inner-join the two legs by video id and emit per-pair deltas.

    Returns ``(deltas, baseline_only_ids, trained_only_ids)``. The two
    *_only lists are the ids that appeared on exactly one leg; the
    caller surfaces them in the manifest so the operator can audit
    coverage. They are NOT silently dropped in the sense of "the user
    won't know"; they are dropped from the bootstrap with a manifest
    record.
    """
    by_id_b = {r.video_id: r for r in baseline}
    by_id_t = {r.video_id: r for r in trained}
    common = sorted(set(by_id_b) & set(by_id_t))
    baseline_only = sorted(set(by_id_b) - set(by_id_t))
    trained_only = sorted(set(by_id_t) - set(by_id_b))

    deltas: list[PairedDelta] = []
    for vid in common:
        b = by_id_b[vid]
        tr = by_id_t[vid]
        composite_delta = tr.composite - b.composite

        secondary_axes = sorted(set(b.secondary_axes) & set(tr.secondary_axes))
        # If a secondary axis appeared on one leg but not the other,
        # the load step should have rejected it; guard anyway.
        if set(b.secondary_axes) != set(tr.secondary_axes):
            raise JudgeAxisMissingError(
                f"pair {vid!r} has asymmetric secondary axes "
                f"(baseline={sorted(b.secondary_axes)}, trained={sorted(tr.secondary_axes)})"
            )
        secondary_deltas = {
            ax: tr.secondary_axes[ax] - b.secondary_axes[ax]
            for ax in secondary_axes
        }
        deltas.append(PairedDelta(
            video_id=vid,
            composite_delta=composite_delta,
            secondary_deltas=secondary_deltas,
        ))

    return deltas, baseline_only, trained_only


# --- bootstrap --------------------------------------------------------------

def _resolve_bootstrap_seed(seed_value: str | int) -> int:
    """Hash the seed string to a 64-bit int; pass through if already int."""
    if isinstance(seed_value, int):
        return seed_value & ((1 << 64) - 1)
    h = hashlib.sha256(str(seed_value).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def paired_bootstrap_ci(
    deltas: list[float],
    n_iters: int,
    seed_value: str | int,
    alpha: float = 0.05,
) -> BootstrapCI:
    """Mean / std / 95% percentile CI from a paired-resample bootstrap.

    ``deltas`` is the per-pair delta vector (composite or one secondary
    axis); resampling is at the *pair index* level, with replacement.
    Caller MUST pass per-pair deltas, not raw per-leg scores — this
    function does not know about legs and cannot enforce pairing
    correctness; that contract lives in :func:`pair_records_by_video_id`.

    Uses ``random.Random`` rather than NumPy to keep the dependency
    footprint at the standard library; for ``n_iters=10000`` and
    ``len(deltas)=42`` this completes in well under a second on any
    modern CPU.
    """
    import random
    import statistics

    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")
    if len(deltas) < 2:
        raise ValueError(
            f"need >=2 pairs to bootstrap, got {len(deltas)}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    rng = random.Random(_resolve_bootstrap_seed(seed_value))
    n = len(deltas)

    sample_means: list[float] = []
    for _ in range(n_iters):
        # Draw n pair-indices with replacement from {0..n-1} and average.
        s = 0.0
        for _i in range(n):
            s += deltas[rng.randrange(n)]
        sample_means.append(s / n)

    sample_means.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_iters))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_iters)) - 1
    lo_idx = max(0, min(n_iters - 1, lo_idx))
    hi_idx = max(0, min(n_iters - 1, hi_idx))

    return BootstrapCI(
        mean=statistics.fmean(deltas),
        std=statistics.pstdev(deltas) if n >= 2 else 0.0,
        ci_low=sample_means[lo_idx],
        ci_high=sample_means[hi_idx],
    )


def bootstrap_all_axes(
    deltas: list[PairedDelta],
    n_iters: int,
    seed_value: str | int,
) -> tuple[BootstrapCI, dict[str, BootstrapCI]]:
    """Run the paired bootstrap for the composite + each secondary axis."""
    composite = paired_bootstrap_ci(
        [d.composite_delta for d in deltas],
        n_iters=n_iters,
        seed_value=seed_value,
    )
    if not deltas:
        return composite, {}
    secondary_keys = sorted(deltas[0].secondary_deltas.keys())
    secondary: dict[str, BootstrapCI] = {}
    for key in secondary_keys:
        # Re-derive the seed per axis so each axis CI is independently
        # reproducible without sharing RNG state across axes.
        per_axis_seed = f"{seed_value}::{key}"
        secondary[key] = paired_bootstrap_ci(
            [d.secondary_deltas[key] for d in deltas],
            n_iters=n_iters,
            seed_value=per_axis_seed,
        )
    return composite, secondary


# --- harness invocation (skeleton) ------------------------------------------

def run_full_judge_eval(
    harness_script: pathlib.Path,
    leg_label: str,
    out_dir: pathlib.Path,
    extra_eval_args: list[str],
) -> pathlib.Path:
    """Invoke ``serveandeval_9b.sh`` for one leg and return the eval-JSON path.

    Round-1 skeleton wrapper. The harness script already starts a vLLM
    server, runs ``run_eval.py`` against it, and writes the eval JSON
    under ``data/scores/ourckpt/`` per the harness defaults; this
    wrapper just shells out and surfaces the chosen save path.

    The wrapper is exercised end-to-end only when ``--actually-run-judge``
    is passed; the round-1 commit's CI does not invoke this code path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_path = out_dir / f"eval_phyjudge_9b__{leg_label}__{timestamp}.json"

    cmd = ["bash", str(harness_script), *extra_eval_args, "--save_path", str(save_path)]
    print(f"=== running PhyJudge harness for leg={leg_label!r}: {' '.join(cmd)} ===")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"PhyJudge harness exited with status {completed.returncode} "
            f"for leg {leg_label!r}; see harness stderr above"
        )
    if not save_path.exists():
        raise RuntimeError(
            f"PhyJudge harness completed but produced no JSON at {save_path}"
        )
    return save_path


# --- manifest ---------------------------------------------------------------

def _git_commit_id_or_none(repo_dir: pathlib.Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _machine_ip_tail_or_none() -> str | None:
    """Return the last octet of the machine's primary IPv4 address.

    Mirrors the convention used by the trainer's run-manifest writer
    (per CLAUDE.md ``machinedoing.md`` updates). Best-effort; returns
    None if no non-loopback IPv4 is found.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # No traffic actually sent; the kernel just resolves the
            # outbound interface for the route.
            s.settimeout(0.2)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        return None
    if not isinstance(ip, str) or "." not in ip:
        return None
    return ip.rsplit(".", 1)[-1]


def _hash_generation_config(generation_config: dict) -> str:
    canonical = json.dumps(generation_config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def stamp_manifest(
    *,
    field_names: list[str],
    axis_mapping: AxisMapping,
    secondary_axis_keys_list: list[str],
    composite_ci: BootstrapCI,
    secondary_cis: dict[str, BootstrapCI],
    n_pairs: int,
    baseline_only_ids: list[str],
    trained_only_ids: list[str],
    bootstrap_iters: int,
    bootstrap_seed: str,
    baseline_ckpt_path: str,
    trained_ckpt_path: str,
    generation_config: dict,
    comparator_pair: str,
    compute_envelope: str,
    commit_id: str | None,
    machine_ip_tail: str | None,
    failure: dict | None = None,
) -> dict:
    """Assemble the run manifest as a plain dict; caller serializes."""
    return {
        "schema_version": 1,
        "kind": "phyjudge_paired_delta_manifest",
        "recipe_id": KNOWN_GOOD_RECIPE_ID,
        "aggregation_rule": AGGREGATION_RULE,
        "comparator_pair": comparator_pair,
        "compute_envelope": compute_envelope,
        "commit_id": commit_id,
        "machine_ip_tail": machine_ip_tail,
        "judge": {
            "harness": "phyjudge_9b",
            "field_names": field_names,
            "axis_mapping": axis_mapping.to_serializable(),
            "secondary_axis_keys": list(secondary_axis_keys_list),
            "composite_definition": "+".join(COMPOSITE_AXES),
        },
        "checkpoints": {
            "baseline_ckpt_path": baseline_ckpt_path,
            "trained_ckpt_path": trained_ckpt_path,
        },
        "generation_config": {
            "sha256_short": _hash_generation_config(generation_config),
            "fields": generation_config,
        },
        "bootstrap": {
            "iters": bootstrap_iters,
            "seed": bootstrap_seed,
            "alpha": 0.05,
            "n_pairs_used": n_pairs,
            "pair_join_unmatched": {
                "baseline_only_ids": baseline_only_ids,
                "trained_only_ids": trained_only_ids,
            },
        },
        "results": {
            "primary_composite": composite_ci.to_serializable(),
            "secondary_axes": {
                axis: ci.to_serializable() for axis, ci in secondary_cis.items()
            },
        },
        "failure": failure,
        "produced_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


# --- markdown writer --------------------------------------------------------

_MD_TEMPLATE = """\
# wan2.2-i2v-A14B v0 i2v-orig-init DPO — PhyJudge-9B Paired Delta

> Terminal scoring report for the v0 i2v-original-init DPO run on the
> 42-prompt heldout split. Comparator pair: **{comparator_pair}**.
> Recipe id ``{recipe_id}`` (frozen). Aggregation rule
> ``{aggregation_rule}`` (frozen). Bootstrap iters ``{bootstrap_iters}``,
> seed ``{bootstrap_seed}``, paired-by-prompt at the pair-index level.

## Comparator

| Leg | Checkpoint path |
|-----|-----------------|
| baseline (original-init no-DPO) | ``{baseline_ckpt_path}`` |
| trained (original-init DPO) | ``{trained_ckpt_path}`` |

## Provenance

| Field | Value |
|-------|-------|
| code commit id | ``{commit_id}`` |
| machine internal-IP tail | ``{machine_ip_tail}`` |
| compute envelope | ``{compute_envelope}`` |
| generation_config sha256[:16] | ``{gen_config_sha}`` |
| pairs used | {n_pairs} |
| baseline-only ids (excluded) | {baseline_only_count} |
| trained-only ids (excluded) | {trained_only_count} |
| produced at (UTC) | ``{produced_at}`` |

## Judge probe

Probed field-name list (top-level ∪ per-result, sorted):

```
{field_names_block}
```

Axis mapping (semantic → probed key):

| Semantic axis | Probed key |
|---------------|-----------|
{axis_mapping_rows}

## Primary composite delta — ``SA + PTV + persistence``

| mean | std | 95% CI low | 95% CI high |
|------|-----|------------|-------------|
| {comp_mean:.4f} | {comp_std:.4f} | {comp_lo:.4f} | {comp_hi:.4f} |

## Secondary axes

{secondary_block}

## Failure record

``{failure_block}``
"""


def _format_secondary_block(secondary_cis: dict[str, BootstrapCI]) -> str:
    if not secondary_cis:
        return "*(no secondary axes returned by probe)*"
    lines = ["| axis | mean | std | 95% CI low | 95% CI high |",
             "|------|------|-----|------------|-------------|"]
    for axis in sorted(secondary_cis.keys()):
        ci = secondary_cis[axis]
        lines.append(
            f"| {axis} | {ci.mean:.4f} | {ci.std:.4f} | "
            f"{ci.ci_low:.4f} | {ci.ci_high:.4f} |"
        )
    return "\n".join(lines)


def write_results_md(
    out_path: pathlib.Path,
    manifest: dict,
) -> None:
    """Render the manifest into the human-facing experiment-results md.

    The md is the primary human-facing artifact; the manifest JSON is the
    machine-facing one. Drift between the two is treated as a bug.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    j = manifest["judge"]
    g = manifest["generation_config"]
    b = manifest["bootstrap"]
    r = manifest["results"]
    primary = r["primary_composite"]

    field_names_block = "\n".join(j["field_names"])
    axis_mapping_rows = "\n".join(
        f"| {ax} | ``{j['axis_mapping'][ax]}`` |" for ax in COMPOSITE_AXES
    )
    secondary_block = _format_secondary_block({
        k: BootstrapCI(**v) for k, v in r["secondary_axes"].items()
    })
    failure_block = json.dumps(manifest.get("failure"), sort_keys=True) \
        if manifest.get("failure") else "null (no halt)"

    text = _MD_TEMPLATE.format(
        comparator_pair=manifest["comparator_pair"],
        recipe_id=manifest["recipe_id"],
        aggregation_rule=manifest["aggregation_rule"],
        bootstrap_iters=b["iters"],
        bootstrap_seed=b["seed"],
        baseline_ckpt_path=manifest["checkpoints"]["baseline_ckpt_path"],
        trained_ckpt_path=manifest["checkpoints"]["trained_ckpt_path"],
        commit_id=manifest.get("commit_id") or "unknown",
        machine_ip_tail=manifest.get("machine_ip_tail") or "unknown",
        compute_envelope=manifest["compute_envelope"],
        gen_config_sha=g["sha256_short"],
        n_pairs=b["n_pairs_used"],
        baseline_only_count=len(b["pair_join_unmatched"]["baseline_only_ids"]),
        trained_only_count=len(b["pair_join_unmatched"]["trained_only_ids"]),
        produced_at=manifest["produced_at_utc"],
        field_names_block=field_names_block,
        axis_mapping_rows=axis_mapping_rows,
        comp_mean=primary["mean"],
        comp_std=primary["std"],
        comp_lo=primary["ci_low"],
        comp_hi=primary["ci_high"],
        secondary_block=secondary_block,
        failure_block=failure_block,
    )
    out_path.write_text(text)


# --- CLI --------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PhyJudge-9B paired-delta scoring (v0 i2v-orig-init DPO).",
    )
    p.add_argument("--probe-results-json", type=pathlib.Path, required=True,
                   help="Path to a probe-eval JSON; this script does NOT "
                        "invoke the harness for the probe step (run "
                        "serveandeval_9b.sh --limit 1 separately first).")
    p.add_argument("--baseline-results-json", type=pathlib.Path, required=True,
                   help="Pre-existing eval JSON for the baseline leg "
                        "(saved by serveandeval_9b.sh + run_eval.py).")
    p.add_argument("--trained-results-json", type=pathlib.Path, required=True,
                   help="Pre-existing eval JSON for the trained leg.")
    p.add_argument("--axis-mapping-json", type=pathlib.Path, default=None,
                   help="Optional JSON map {semantic_axis: probed_key} "
                        "for renaming when the probe spelling differs.")
    p.add_argument("--baseline-ckpt-path", type=str, required=True)
    p.add_argument("--trained-ckpt-path", type=str, required=True)
    p.add_argument("--generation-config-json", type=pathlib.Path, required=True,
                   help="Generation config used to produce the heldout "
                        "videos for both legs (must be byte-identical "
                        "across legs by the encode-time contract).")
    p.add_argument("--out-manifest", type=pathlib.Path, required=True)
    p.add_argument("--out-md", type=pathlib.Path, required=True)
    p.add_argument("--bootstrap-iters", type=int, default=DEFAULT_BOOTSTRAP_ITERS,
                   help="Default 10000 is well past the regime where the "
                        "95%% CI shifts more than score-discretization "
                        "granularity for 42-pair samples.")
    p.add_argument("--bootstrap-seed", type=str, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--comparator-pair", type=str,
                   default="original-init no-DPO baseline vs original-init DPO trained")
    p.add_argument("--compute-envelope", type=str, default="single_gpu",
                   choices=("single_gpu", "dpo_multi_gpu_zero2"),
                   help="Stamped into the manifest per-run. PhyJudge-9B "
                        "serving fits on one A100 80GB so 'single_gpu' is "
                        "the typical envelope; 'dpo_multi_gpu_zero2' is "
                        "kept as a stamp option for symmetry with trainer "
                        "manifests but does not change judge behaviour.")
    p.add_argument("--commit-id", type=str, default=None,
                   help="Code commit id; auto-detected via 'git rev-parse "
                        "HEAD' from the script's directory if absent.")
    p.add_argument("--machine-ip-tail", type=str, default=None,
                   help="Machine internal-IP last octet; auto-detected if "
                        "absent.")
    p.add_argument("--actually-run-judge", action="store_true",
                   help="Reserved for the M6→M8 hand-off. Skeleton commit "
                        "ignores this flag; the harness invocation lives "
                        "in run_full_judge_eval and will be wired into "
                        "main() once M6 lands the heldout videos.")
    return p


def _write_failure_manifest(
    out_manifest: pathlib.Path,
    failure_kind: str,
    failure_message: str,
    extra: dict | None = None,
) -> None:
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "phyjudge_paired_delta_manifest",
        "failure": {
            "kind": failure_kind,
            "message": failure_message,
            "extra": extra or {},
        },
        "produced_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    out_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    axis_mapping_override: dict[str, str] | None = None
    if args.axis_mapping_json is not None:
        axis_mapping_override = json.loads(args.axis_mapping_json.read_text())
        if not isinstance(axis_mapping_override, dict):
            print(f"--axis-mapping-json must be a JSON object", file=sys.stderr)
            return 2

    # Step 1 — probe.
    try:
        field_names, axis_mapping = probe_phyjudge_axes(
            args.probe_results_json,
            axis_mapping_override=axis_mapping_override,
        )
    except JudgeAxisMissingError as exc:
        _write_failure_manifest(
            args.out_manifest,
            failure_kind="judge-axis-missing",
            failure_message=str(exc),
            extra={"step": "probe"},
        )
        print(f"HALT judge-axis-missing (probe): {exc}", file=sys.stderr)
        return 3

    # Step 1b — eligible secondary axes from the same probe sample.
    probe_payload = json.loads(args.probe_results_json.read_text())
    secondary_keys = secondary_axis_keys(
        probe_payload["results"][0],
        axis_mapping,
    )

    # Step 2 — load both legs against the resolved mapping + secondary set.
    try:
        baseline = load_eval_results(
            args.baseline_results_json, axis_mapping, secondary_keys,
        )
        trained = load_eval_results(
            args.trained_results_json, axis_mapping, secondary_keys,
        )
    except JudgeAxisMissingError as exc:
        _write_failure_manifest(
            args.out_manifest,
            failure_kind="judge-axis-missing",
            failure_message=str(exc),
            extra={"step": "load_eval_results"},
        )
        print(f"HALT judge-axis-missing (load): {exc}", file=sys.stderr)
        return 3

    # Step 3 — pair + bootstrap.
    deltas, baseline_only, trained_only = pair_records_by_video_id(baseline, trained)
    if len(deltas) < 2:
        _write_failure_manifest(
            args.out_manifest,
            failure_kind="insufficient-pairs",
            failure_message=(
                f"only {len(deltas)} pairs survived the inner-join "
                f"(baseline-only={len(baseline_only)}, trained-only={len(trained_only)}); "
                f"need at least 2 to bootstrap"
            ),
            extra={"step": "pair_records_by_video_id"},
        )
        print(f"HALT insufficient-pairs: {len(deltas)} pairs", file=sys.stderr)
        return 4

    composite_ci, secondary_cis = bootstrap_all_axes(
        deltas,
        n_iters=args.bootstrap_iters,
        seed_value=args.bootstrap_seed,
    )

    # Step 4 — manifest + md.
    generation_config = json.loads(args.generation_config_json.read_text())
    if not isinstance(generation_config, dict):
        print(f"--generation-config-json must be a JSON object", file=sys.stderr)
        return 2

    commit_id = args.commit_id or _git_commit_id_or_none(
        pathlib.Path(__file__).resolve().parent
    )
    machine_ip_tail = args.machine_ip_tail or _machine_ip_tail_or_none()

    manifest = stamp_manifest(
        field_names=field_names,
        axis_mapping=axis_mapping,
        secondary_axis_keys_list=secondary_keys,
        composite_ci=composite_ci,
        secondary_cis=secondary_cis,
        n_pairs=len(deltas),
        baseline_only_ids=baseline_only,
        trained_only_ids=trained_only,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        baseline_ckpt_path=args.baseline_ckpt_path,
        trained_ckpt_path=args.trained_ckpt_path,
        generation_config=generation_config,
        comparator_pair=args.comparator_pair,
        compute_envelope=args.compute_envelope,
        commit_id=commit_id,
        machine_ip_tail=machine_ip_tail,
        failure=None,
    )
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_results_md(args.out_md, manifest)

    print(f"OK: {len(deltas)} pairs scored")
    print(f"  composite Δ mean = {composite_ci.mean:.4f}, "
          f"std = {composite_ci.std:.4f}, "
          f"95% CI = [{composite_ci.ci_low:.4f}, {composite_ci.ci_high:.4f}]")
    for axis in sorted(secondary_cis):
        ci = secondary_cis[axis]
        print(f"  {axis:24s} Δ mean = {ci.mean:.4f}, "
              f"std = {ci.std:.4f}, "
              f"95% CI = [{ci.ci_low:.4f}, {ci.ci_high:.4f}]")
    print(f"  manifest: {args.out_manifest}")
    print(f"  results md: {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
