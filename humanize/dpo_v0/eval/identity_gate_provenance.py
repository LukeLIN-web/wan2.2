"""AC-2.1 provenance-only identity gate.

Reads the policy and reference manifests produced by ``canonical_loader``
and asserts ``policy_base_merged_sha256 == reference_merged_sha256`` plus
per-key sidecar agreement. The numerical (AC-2.2) and step-0 generation
parity (AC-2.3) checks require the full WanModel forward path (T5 + I2V
conditioning + scheduler) and are deferred to the trainer-integration
sub-step in the morning supervised run.

This script is the trivial-but-honest part of M2: provenance is the
weakest of the three identity-gate levels but it's what's verifiable
without the full pipeline. It runs in seconds and produces an explicit
PASS / FAIL line plus a stamped report.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys


def per_key_sidecar_agreement(side_a: pathlib.Path, side_b: pathlib.Path) -> tuple[bool, int, list[dict]]:
    """Stream both per-key JSONLs in lockstep, comparing every line.

    Returns (agree, tensor_count, mismatches) where mismatches is a
    list of the first 10 mismatched lines (key + side_a + side_b).
    Following AC-2.1's "per-key hashes are never loaded into memory in
    aggregate" guidance, lines are compared one at a time.
    """
    mismatches: list[dict] = []
    n = 0
    with side_a.open("rb") as fa, side_b.open("rb") as fb:
        for ln_a, ln_b in zip(fa, fb):
            n += 1
            if ln_a != ln_b:
                if len(mismatches) < 10:
                    try:
                        ja = json.loads(ln_a)
                        jb = json.loads(ln_b)
                    except json.JSONDecodeError:
                        ja = {"raw": ln_a.decode("utf-8", "replace")}
                        jb = {"raw": ln_b.decode("utf-8", "replace")}
                    mismatches.append({"line": n, "a": ja, "b": jb})
        # ensure both files exhausted
        leftover_a = fa.read()
        leftover_b = fb.read()
        if leftover_a or leftover_b:
            mismatches.append({"line": n + 1, "a_leftover_bytes": len(leftover_a), "b_leftover_bytes": len(leftover_b)})
    return len(mismatches) == 0, n, mismatches


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-manifest", type=pathlib.Path, required=True)
    ap.add_argument("--reference-manifest", type=pathlib.Path, required=True)
    ap.add_argument("--report-out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv[1:])

    policy = json.loads(args.policy_manifest.read_bytes())
    reference = json.loads(args.reference_manifest.read_bytes())

    policy_sha = policy["merged_state_sha256"]
    reference_sha = reference["merged_state_sha256"]
    sha_match = policy_sha == reference_sha
    sidecar_match = policy["sidecar_jsonl_sha256"] == reference["sidecar_jsonl_sha256"]
    side_a = pathlib.Path(policy["sidecar_jsonl_path"])
    side_b = pathlib.Path(reference["sidecar_jsonl_path"])
    agree, n_lines, mismatches = per_key_sidecar_agreement(side_a, side_b)

    verdict = "PASS" if sha_match and sidecar_match and agree else "FAIL"
    report = {
        "verdict": verdict,
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy_manifest": str(args.policy_manifest),
        "reference_manifest": str(args.reference_manifest),
        "policy_merged_state_sha256": policy_sha,
        "reference_merged_state_sha256": reference_sha,
        "merged_sha_match": sha_match,
        "sidecar_jsonl_sha_match": sidecar_match,
        "per_key_sidecar_streaming_match": agree,
        "tensor_count_streamed": n_lines,
        "policy_recipe_id": policy["recipe_id"],
        "reference_recipe_id": reference["recipe_id"],
        "first_mismatches": mismatches,
        "ac_2_1_status": "PASS" if (sha_match and sidecar_match and agree) else "FAIL",
        "ac_2_2_status": "deferred-to-morning-supervised-run",
        "ac_2_3_status": "deferred-to-morning-supervised-run",
        "notes": (
            "AC-2.1 provenance verified by stamped manifest hashes + streaming "
            "per-key sidecar agreement. AC-2.2 (numerical noise-pred allclose) "
            "and AC-2.3 (step-0 generation parity per-frame L1/SSIM/PSNR) "
            "require the full WanModel forward path (T5 + I2V conditioning + "
            "scheduler) and the trainer integration; deferred to morning luke1 "
            "supervised run per round-2-summary handoff."
        ),
    }

    out_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(out_text, encoding="utf-8")
    print(out_text)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
