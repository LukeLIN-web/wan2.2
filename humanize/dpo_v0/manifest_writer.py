"""Run-manifest schema and writer for DPO-on-Wan2.2-I2V-A14B v0 runs.

Stamps every training / generation / evaluation run with a deterministic
provenance record so that downstream consumers (eval harness, paired-delta
report, codex review) can recover the full identity of the run from one
JSON file.

Plan reference: ``videodpo:humanize/i2v.md`` AC-3 (pin contract), AC-5.U4
(routing counter logging), AC-6 (compute envelope + ref-offload semantics),
AC-8 (terminal report fields). The 16-hex frozen ``recipe_id`` value is
asserted via :func:`assert_recipe_pins`; see plan AC-3.1 for derivation.

Hard contracts surfaced in this module
--------------------------------------
* The on-disk pin is treated as authority **only after** it is cross-checked
  against the recompute of ``sha256(canonical_yaml.bytes).hex[:16]`` and
  against the ``KNOWN_GOOD_RECIPE_ID`` constant. The constant is
  defense-in-depth, not authority.
* Per-key tensor SHAs are emitted to a sidecar JSONL and never inlined into
  the manifest. The manifest stamps ``merged_state_sha256``, the sidecar
  path, the sidecar's own SHA256, and the tensor count -- nothing more.
* All file writes are atomic (temp file + rename). A partial manifest is an
  abort signal for downstream consumers.
* When the judge probe is missing one of ``{SA, PTV, persistence}``, the
  caller is required to invoke :func:`halt_judge_axis_missing` rather than
  silently substituting another axis. The caller writes a permanent
  ``judge_axis_missing.json`` audit file alongside whatever manifest state
  was already on disk.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib
import socket
import subprocess
from typing import Any, Iterable, Iterator

import torch

# ---------------------------------------------------------------------------
# Frozen pins (see plan AC-3 for derivation; do not edit without a plan
# amendment commit citing the new value).
# ---------------------------------------------------------------------------

#: Frozen recipe ID; equals ``sha256(canonical_yaml.bytes).hex[:16]`` of
#: ``humanize/dpo_v0/recipes/wan22_i2v_a14b__round2_v0.yaml`` under the
#: canonical serializer rules. See plan AC-3.1.
KNOWN_GOOD_RECIPE_ID = "6bef6e104cdd3442"

#: Recipe yaml filename and on-disk pin filename. Live next to each other
#: under ``<recipes_dir>/`` so that re-reading and re-hashing is one
#: directory walk.
RECIPE_YAML_FILENAME = "wan22_i2v_a14b__round2_v0.yaml"
RECIPE_ID_FILENAME = "recipe_id"

#: Scheduler / data axes that must appear in the run manifest's
#: ``recipe_pins`` block. See plan AC-3.5.
EXPECTED_SWITCH_DIT_BOUNDARY = 0.9
EXPECTED_FPS = 16
EXPECTED_FRAME_NUM = 81
EXPECTED_AGGREGATION_RULE = "cross_group_rater_union"
EXPECTED_DTYPE_POLICY = "bf16_forward_fp32_master"

#: Canonical compute-envelope enum. The set is documented in plan AC-6 /
#: DEC-6. ``dpo_multi_gpu_ddp`` is the in-flight honest-enum amendment from
#: round-3 trainer integration (see plan amendment commit
#: ``videodpo@a52fb90`` and the rl2 ``df979b3d`` rationale); the set here
#: tolerates it without rejecting so a trainer can stamp the value it
#: actually ran under.
COMPUTE_ENVELOPES_CANONICAL = (
    "single_gpu",
    "dpo_multi_gpu_zero2",
    "dpo_multi_gpu_ddp",
    "multi_gpu_inference_seed_parallel",
)

#: PhyJudge axes that the trainer / eval probe must surface. Plan DEC-7 /
#: AC-8 require composite ``SA + PTV + persistence``; absence of any one
#: triggers :func:`halt_judge_axis_missing` rather than silent substitution.
JUDGE_REQUIRED_AXES = ("SA", "PTV", "persistence")

#: Streaming canonical hash field separator. See plan AC-3.4.
FIELD_SEPARATOR = b"|"

#: Manifest schema version. Bump on any incompatible field change so that a
#: stale reader fails loudly instead of misinterpreting.
MANIFEST_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _file_sha256(path: pathlib.Path, buf: int = 4 * 1024 * 1024) -> str:
    """Streaming SHA-256 of a single file path; bounded memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_iso_timestamp() -> str:
    """UTC ISO 8601 timestamp with explicit ``+00:00`` offset."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git_commit_id(repo_dir: pathlib.Path | None = None) -> str:
    """Return the full ``git rev-parse HEAD`` of ``repo_dir`` (or cwd).

    Bubbles the ``CalledProcessError`` so a missing repo / detached state
    surfaces immediately rather than stamping a stale or empty value.
    """
    cmd = ["git", "rev-parse", "HEAD"]
    cwd = str(repo_dir) if repo_dir is not None else None
    out = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def machine_internal_ip_tail() -> str:
    """Return the last octet of the machine's internal IPv4 address.

    Used as a short, human-readable identifier in the run manifest so that
    cross-box rsync / re-deploy provenance is recoverable from the manifest
    alone (e.g. ``juyi-finetune`` → ``.196``). Returns ``".0"`` if the
    socket lookup fails -- this is intentionally non-fatal because some
    eval-only jobs run inside containers without internal-IP awareness.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            return "." + ip.rsplit(".", 1)[-1]
    except OSError:
        return ".0"


def atomic_write_text(path: pathlib.Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` via ``path.tmp`` rename.

    The temp file lives next to the destination so that the ``rename`` is
    atomic on the same filesystem. A partial write leaves the temp file
    behind for forensics but never replaces the destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_json(path: pathlib.Path, obj: Any, *, indent: int = 2) -> None:
    """Atomic JSON write with deterministic key order."""
    atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=True))


# ---------------------------------------------------------------------------
# Recipe pin assertion
# ---------------------------------------------------------------------------


def assert_recipe_pins(
    recipes_dir: pathlib.Path,
    *,
    expected_recipe_id: str = KNOWN_GOOD_RECIPE_ID,
) -> dict[str, str]:
    """Read + recompute + assert the on-disk recipe pin.

    Three-way assert before any side-effect (cf. BL-20260428-canonical-recipe-
    pin-runtime-reread):

    1. Read the on-disk pin from ``<recipes_dir>/recipe_id``.
    2. Recompute ``sha256(canonical_yaml.bytes).hex[:16]`` over the bytes of
       ``<recipes_dir>/wan22_i2v_a14b__round2_v0.yaml``.
    3. Assert (1) == (2) == ``expected_recipe_id``.

    Returns a dict of pin values suitable to splat into a manifest's
    ``recipe_pins`` block. The caller is expected to layer in
    ``vae_sha256`` / ``t5_sha256`` / ``tokenizer_tree_sha256`` once those
    hot-path helpers land (see plan AC-3.2 / AC-3.3 contract).

    Raises ``AssertionError`` with all three values inline so that operator
    eyeballs can spot which side drifted.
    """
    yaml_bytes = (recipes_dir / RECIPE_YAML_FILENAME).read_bytes()
    fresh = hashlib.sha256(yaml_bytes).hexdigest()[:16]
    on_disk = (recipes_dir / RECIPE_ID_FILENAME).read_text(encoding="ascii").strip()
    assert fresh == on_disk == expected_recipe_id, (
        f"recipe pin drift: fresh={fresh}, on_disk={on_disk}, expected={expected_recipe_id}"
    )
    return {
        "recipe_id": on_disk,
        "switch_DiT_boundary": EXPECTED_SWITCH_DIT_BOUNDARY,
        "fps": EXPECTED_FPS,
        "frame_num": EXPECTED_FRAME_NUM,
        "aggregation_rule": EXPECTED_AGGREGATION_RULE,
        "dtype_policy": EXPECTED_DTYPE_POLICY,
    }


# ---------------------------------------------------------------------------
# Streaming canonical hash (sidecar JSONL emitter)
# ---------------------------------------------------------------------------


def streaming_canonical_hash(
    tensor_iter: Iterable[tuple[str, torch.Tensor]],
    sidecar_path: pathlib.Path,
) -> tuple[str, str, int]:
    """Walk ``(key, tensor)`` pairs and emit a sidecar JSONL.

    Caller MUST yield pairs in alphabetical key order; this function does
    not sort. Each pair contributes to the merged hasher and writes one
    JSON line ``{"key", "shape", "dtype", "sha256"}`` to ``sidecar_path``
    using a freshly-zeroed per-key hasher. Tensor bytes are released after
    the hasher updates, so peak memory is bounded by one tensor.

    Returns ``(merged_state_sha256, sidecar_sha256, tensor_count)``. The
    sidecar SHA-256 is computed by re-reading the file after closing it,
    which keeps the on-disk artifact and the stamped hash in lockstep
    even if a buggy callsite updates the sidecar after the fact (the
    re-read forces honest agreement).

    The byte spectrum (header field order, separator, ``str(dtype)`` and
    ``repr(tuple(shape))`` form) is documented in
    ``loader/MANIFEST_SCHEMA.md`` and tested in
    ``loader/test_canonical_loader.py``. This module reuses the same byte
    spectrum so a manifest produced here is comparable to a loader-emitted
    manifest under the canonical hash rule.
    """
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    merged = hashlib.sha256()
    tensor_count = 0
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for key, tensor in tensor_iter:
            shape_repr = repr(tuple(tensor.shape))
            dtype_repr = str(tensor.dtype)
            tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()

            per_key = hashlib.sha256()
            for hasher in (merged, per_key):
                hasher.update(key.encode("utf-8"))
                hasher.update(FIELD_SEPARATOR)
                hasher.update(shape_repr.encode("utf-8"))
                hasher.update(FIELD_SEPARATOR)
                hasher.update(dtype_repr.encode("utf-8"))
                hasher.update(FIELD_SEPARATOR)
                hasher.update(tensor_bytes)

            f.write(
                json.dumps(
                    {
                        "key": key,
                        "shape": list(tensor.shape),
                        "dtype": dtype_repr,
                        "sha256": per_key.hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            tensor_count += 1
            del tensor_bytes
    tmp.replace(sidecar_path)
    sidecar_sha = _file_sha256(sidecar_path)
    return merged.hexdigest(), sidecar_sha, tensor_count


# ---------------------------------------------------------------------------
# Run manifest schema
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ShardEntry:
    """One row of the ordered shard manifest. See plan AC-1."""

    file: str
    sha256: str
    param_count: int
    dtype: str


@dataclasses.dataclass
class CkptSourcePaths:
    """Where the policy / reference / LoRA weights came from on disk."""

    high_noise_base: str
    low_noise_frozen: str
    lora_adapter: str | None = None


@dataclasses.dataclass
class RoutingForwardEntry:
    """One forward-pass routing log entry. See plan AC-5.U4."""

    sampled_timestep_id: int
    raw_timestep: int
    detected_expert: str  # "high_noise" | "low_noise"


@dataclasses.dataclass
class JudgeFieldProbe:
    """Result of probing the PhyJudge serve endpoint for axis names.

    ``axis_to_field`` maps the canonical axis name (``"SA"`` / ``"PTV"`` /
    ``"persistence"``) to the JSON field-name returned by the probe; the
    composite primary scalar is computed as the sum of these three. If the
    probe was missing any required axis, ``axis_to_field`` will not contain
    that key and ``halt_judge_axis_missing`` should be invoked.
    """

    axis_to_field: dict[str, str]
    raw_probe_payload: dict[str, Any]


@dataclasses.dataclass
class RunManifest:
    """Provenance record for one run.

    All fields except ``commit_id`` / ``timestamp`` are caller-supplied.
    Validated by :meth:`__post_init__` (lightweight: required-field presence,
    no schema-completeness because this dataclass is the schema). The
    ``compute_envelope`` value is checked against
    :data:`COMPUTE_ENVELOPES_CANONICAL` for typo defence; an unknown value
    raises rather than silently stamping garbage.
    """

    timestamp: str
    commit_id: str
    machine_internal_ip_tail: str
    compute_envelope: str
    ckpt_source_paths: CkptSourcePaths
    shard_manifest: list[ShardEntry]
    merged_state_sha256: str
    per_key_sidecar_path: str
    per_key_sidecar_sha256: str
    tensor_count: int
    recipe_pins: dict[str, Any]
    routing_counter_log: list[RoutingForwardEntry]
    judge_field_probe: JudgeFieldProbe | None
    generation_config: dict[str, Any]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.compute_envelope not in COMPUTE_ENVELOPES_CANONICAL:
            raise ValueError(
                f"unknown compute_envelope {self.compute_envelope!r}; "
                f"expected one of {COMPUTE_ENVELOPES_CANONICAL}"
            )
        if self.tensor_count < 0:
            raise ValueError(f"tensor_count must be non-negative, got {self.tensor_count}")
        if not self.commit_id:
            raise ValueError("commit_id is required")
        if not self.timestamp:
            raise ValueError("timestamp is required")
        for required in ("recipe_id", "switch_DiT_boundary", "fps", "frame_num"):
            if required not in self.recipe_pins:
                raise ValueError(f"recipe_pins missing required key {required!r}")

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-friendly dict with deterministic ordering."""
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "commit_id": self.commit_id,
            "machine_internal_ip_tail": self.machine_internal_ip_tail,
            "compute_envelope": self.compute_envelope,
            "ckpt_source_paths": dataclasses.asdict(self.ckpt_source_paths),
            "shard_manifest": [dataclasses.asdict(s) for s in self.shard_manifest],
            "merged_state_sha256": self.merged_state_sha256,
            "per_key_sidecar_path": self.per_key_sidecar_path,
            "per_key_sidecar_sha256": self.per_key_sidecar_sha256,
            "tensor_count": self.tensor_count,
            "recipe_pins": self.recipe_pins,
            "routing_counter_log": [dataclasses.asdict(e) for e in self.routing_counter_log],
            "judge_field_probe": (
                dataclasses.asdict(self.judge_field_probe) if self.judge_field_probe else None
            ),
            "generation_config": self.generation_config,
        }


def write_run_manifest(out_dir: pathlib.Path, manifest: RunManifest) -> pathlib.Path:
    """Atomic-write ``manifest.json`` under ``out_dir`` and return the path.

    The write goes through ``manifest.json.tmp`` and is renamed; a partial
    file means abort. The manifest is stamped with deterministic key order
    (``json.dumps(..., sort_keys=True)``) so two manifests of identical
    content are byte-equal.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    atomic_write_json(path, manifest.to_jsonable())
    return path


# ---------------------------------------------------------------------------
# Judge axis halt path
# ---------------------------------------------------------------------------


class JudgeAxisMissingError(RuntimeError):
    """Raised when the PhyJudge probe omits a required composite axis.

    Plan DEC-7 / AC-8 require that the trainer halts rather than silently
    substituting another axis when ``SA`` / ``PTV`` / ``persistence`` are
    not all returned by ``serveandeval_9b.sh``.
    """


def halt_judge_axis_missing(
    missing_axes: list[str],
    out_dir: pathlib.Path,
    *,
    raw_probe_payload: dict[str, Any] | None = None,
) -> None:
    """Write a permanent audit file and raise.

    The file ``<out_dir>/judge_axis_missing.json`` lives outside any
    manifest written by ``write_run_manifest`` so that a later eval pass
    cannot accidentally overwrite the audit record. Calling this function
    is the only sanctioned response to a probe that omits a required axis;
    swallowing the error or substituting another axis is a contract
    violation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "timestamp": utc_iso_timestamp(),
        "missing_axes": list(missing_axes),
        "required_axes": list(JUDGE_REQUIRED_AXES),
        "raw_probe_payload": raw_probe_payload or {},
        "halt_reason": "judge-axis-missing",
    }
    atomic_write_json(out_dir / "judge_axis_missing.json", payload)
    raise JudgeAxisMissingError(
        f"PhyJudge probe missing required axes {missing_axes}; "
        f"see {out_dir / 'judge_axis_missing.json'}"
    )


def assert_judge_axes_present(
    probe_payload: dict[str, Any],
    out_dir: pathlib.Path,
    *,
    axis_to_field: dict[str, str] | None = None,
) -> JudgeFieldProbe:
    """Convenience: build a :class:`JudgeFieldProbe` or halt.

    ``axis_to_field`` is the caller's mapping from canonical axis name
    (one of ``JUDGE_REQUIRED_AXES``) to the field name returned by the
    probe. If any required axis is unmapped (or maps to a name absent
    from ``probe_payload``), :func:`halt_judge_axis_missing` is invoked.
    """
    if axis_to_field is None:
        axis_to_field = {a: a for a in JUDGE_REQUIRED_AXES}
    missing: list[str] = []
    for axis in JUDGE_REQUIRED_AXES:
        field = axis_to_field.get(axis)
        if field is None or field not in probe_payload:
            missing.append(axis)
    if missing:
        halt_judge_axis_missing(missing, out_dir, raw_probe_payload=probe_payload)
    return JudgeFieldProbe(axis_to_field=dict(axis_to_field), raw_probe_payload=probe_payload)


__all__ = [
    "KNOWN_GOOD_RECIPE_ID",
    "EXPECTED_SWITCH_DIT_BOUNDARY",
    "EXPECTED_FPS",
    "EXPECTED_FRAME_NUM",
    "EXPECTED_AGGREGATION_RULE",
    "EXPECTED_DTYPE_POLICY",
    "COMPUTE_ENVELOPES_CANONICAL",
    "JUDGE_REQUIRED_AXES",
    "MANIFEST_SCHEMA_VERSION",
    "ShardEntry",
    "CkptSourcePaths",
    "RoutingForwardEntry",
    "JudgeFieldProbe",
    "RunManifest",
    "JudgeAxisMissingError",
    "assert_recipe_pins",
    "streaming_canonical_hash",
    "write_run_manifest",
    "halt_judge_axis_missing",
    "assert_judge_axes_present",
    "atomic_write_text",
    "atomic_write_json",
    "utc_iso_timestamp",
    "git_commit_id",
    "machine_internal_ip_tail",
]
