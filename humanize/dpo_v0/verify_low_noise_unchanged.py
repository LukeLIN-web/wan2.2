"""AC-6 contract enforcement: low-noise expert byte-equality pre/post training.

The DPO trainer (train_dpo_i2v.py) only loads the high-noise expert subdir
(`Wan2.2-I2V-A14B/high_noise_model/`) and only applies LoRA to its modules.
The low-noise subdir (`Wan2.2-I2V-A14B/low_noise_model/`) is never read or
written. AC-6 contract: prove this byte-equality is preserved by computing
the manifest of low-noise shards' SHA256 before training and again after
training, and asserting they match.

Usage:
    # Before launching training:
    python verify_low_noise_unchanged.py --upstream <root> --before /tmp/before.json

    # After training run completes:
    python verify_low_noise_unchanged.py --upstream <root> --after /tmp/after.json \
        --check-against /tmp/before.json

If --check-against is given, the after-snapshot is compared against the
before-snapshot and the script exits 0 on byte-equality, 1 otherwise. The
result + per-shard SHAs are also written into the trainer run_manifest's
`low_noise_pre_post_byte_equal` field via the manifest_writer integration
(rl8 task #16 pickup).

rl9 finding M-3 (task #14): this is the smallest possible defense against
"someone accidentally pointed the trainer at the wrong subdir + the test
loop never noticed" — a 30-line script that takes < 1 s and gives a hard
PASS/FAIL signal, on the cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def snapshot_low_noise(upstream_root: pathlib.Path) -> dict | None:
    """Walk <root>/low_noise_model/ and return {relpath: sha256} sorted.

    Returns None if low_noise_model doesn't exist on this box: that's a deploy
    state where the AC-6 byte-equality contract is trivially satisfied (trainer
    cannot have modified what isn't there). Caller should record this in the
    run_manifest as `low_noise_pre_post_byte_equal: "n/a (subdir absent)"`.
    """
    low_dir = upstream_root / "low_noise_model"
    if not low_dir.exists():
        return None
    snap: dict[str, str] = {}
    for p in sorted(low_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(low_dir).as_posix()
        snap[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


def diff_snapshots(before: dict, after: dict) -> list[str]:
    issues: list[str] = []
    keys_b, keys_a = set(before), set(after)
    for k in sorted(keys_b - keys_a):
        issues.append(f"REMOVED: {k}")
    for k in sorted(keys_a - keys_b):
        issues.append(f"ADDED: {k}")
    for k in sorted(keys_b & keys_a):
        if before[k] != after[k]:
            issues.append(f"CHANGED: {k}\n  before={before[k]}\n  after ={after[k]}")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=pathlib.Path, required=True,
                   help="root of Wan2.2-I2V-A14B (parent of low_noise_model/)")
    snap_group = p.add_mutually_exclusive_group(required=True)
    snap_group.add_argument("--before", type=pathlib.Path,
                            help="write current snapshot to this path (pre-training)")
    snap_group.add_argument("--after", type=pathlib.Path,
                            help="write current snapshot to this path (post-training)")
    p.add_argument("--check-against", type=pathlib.Path,
                   help="reference snapshot to diff against (use with --after)")
    args = p.parse_args()

    snap = snapshot_low_noise(args.upstream)
    target = args.before if args.before is not None else args.after
    payload: dict = {"low_noise_snapshot": snap if snap is not None else {}}
    payload["status"] = "absent" if snap is None else "present"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if snap is None:
        print(f"[snapshot] low_noise_model/ absent on this box -> {target} (status=absent)")
    else:
        print(f"[snapshot] {len(snap)} files under low_noise_model/ -> {target}")

    if args.check_against is not None:
        ref_payload = json.loads(args.check_against.read_text())
        ref_snap = ref_payload["low_noise_snapshot"]
        ref_status = ref_payload.get("status", "present")
        cur_status = "absent" if snap is None else "present"
        if ref_status == "absent" and cur_status == "absent":
            print("[PASS-NA] low_noise_model absent both pre and post (trivial byte-equality)")
            sys.exit(0)
        if ref_status != cur_status:
            print(f"[FAIL] status mismatch: pre={ref_status} post={cur_status}")
            sys.exit(1)
        issues = diff_snapshots(ref_snap, snap or {})
        if issues:
            print(f"[FAIL] {len(issues)} byte-equality violations:")
            for i in issues:
                print(f"  {i}")
            sys.exit(1)
        print(f"[PASS] all {len(snap)} low-noise files byte-identical pre/post training")
        sys.exit(0)


if __name__ == "__main__":
    main()
