#!/usr/bin/env python3
"""Round-5 warm-start: build the cond-image-present setminus subset.

Goal (per luke decisions 2026-04-30):
- Round-5 = warm-start β=100 lora_final (round-4 winner) on the *remaining*
  tier_b pair_ids that were NOT used in round-4.
- "Remaining" = round-2 raw 2745-pair pool setminus round-4 used 1000-pair
  set (sha pinned `cf5d3e5fd528a3e0`).
- Disk-image filter (round-4 task #19 protocol): drop pair_ids whose
  conditioning image is not present on disk.
- Heldout-disjoint check: enforced upstream in round-2 T3 split (already
  honored by raw 2745 set).

Two cardinality views:
- ``pair_ids_full_setminus``: 1745 pair_ids (raw 2745 - round-4 used 1000)
  -> AUDIT TRAIL only. luke `0ba44aed` ratified it is not the train target.
- ``pair_ids_cond_image_present``: 1202 pair_ids (cond-present 2202 - round-4
  used 1000) -> OFFICIAL round-5 train set. trainer pin chain asserts this
  set's canonical sha256[:16].

Inheritance from round-4:
- ``recipe_id`` is unchanged = ``6bef6e104cdd3442`` (recipe schema does not
  depend on pair selection; only T3-tier composition + manifest schema +
  encoder versions are pinned in recipe).
- canonical pair_ids sha256 = ``"\n".join(pair_ids) + "\n"`` (newline-joined,
  trailing newline) — same algorithm as round-4 ``canonical_pair_ids_sha256``.
- Order = round-4 protocol (sha256(seed_namespace || recipe_id)[:8] hex,
  random.Random.shuffle), so the order is deterministic but not dictionary-
  sorted. The trainer's pair_ids pin must match the order in this file.

Inputs:
- T3_subset.json (round-2 T3 output): provides ``tier_b.pair_ids`` (2745).
  Source of truth: ``juyi-videorl:/home/user1/T0_T3_root/T3_subset.json``.
- T3_round4_tier_b_1k.json (round-4 task #19 output): provides
  ``tier_b_round4_1k.pair_ids`` (1000). Already on shared FS.
- drop_log.json (round-4 task #19 output): provides the
  ``image_path_disk_missing`` 543-pair list, used to drop the same set so
  round-5 is consistent with round-4 disk filter.

Outputs (Option X layout — rl9 ack `99e5989d`: 2 files, exactly one
top-level ``pair_ids`` list across the train-side payload, no ambiguity
for the trainer's detect-single-key walk):

- ``out/round5/<UTC>/T3_round5_warm_official_1202.json`` — official train
  payload. Top-level wrapper ``tier_b_round5_warm_1202.pair_ids`` (1202)
  is the unique ``.pair_ids`` list in this file. Trainer reads this via
  ``--subset-pair-ids-json``.
- ``out/round5/<UTC>/T3_round5_warm_audit_1745.json`` — audit-trail
  payload. Stores the 1745 raw-setminus list under
  ``tier_b_round5_audit_1745.audit_pair_ids`` (intentionally NOT
  ``pair_ids`` so a misconfigured trainer still cannot accidentally read
  this file). Provenance / disk-missing breakdown lives here.
- ``out/round5/<UTC>/recipe_id_pin`` — frozen recipe_id at build time.
- ``out/round5/<UTC>/pair_ids_cond_image_present_sha256_hex16_pin`` —
  trainer-asserted pin (1202 effective, sha over the shuffled list in
  the official file).
- ``out/round5/<UTC>/pair_ids_full_setminus_sha256_hex16_pin`` —
  audit-only pin (1745 nominal).
- ``out/round5/<UTC>/drop_log.json`` — audit log of which 543 disk-
  missing pair_ids were dropped, mirrored from round-4 build for
  reproducibility.

The output dir is under ``out/`` which is gitignored (round-4 sibling).
Commit the script + the recipe pin files in ``recipes/``.
"""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--t3-subset-json", required=True,
                   help="Path to round-2 T3_subset.json (tier_b 2745 source). "
                        "On juyi-videorl: /home/user1/T0_T3_root/T3_subset.json.")
    p.add_argument("--round4-subset-json", required=True,
                   help="Path to round-4 T3_round4_tier_b_1k.json (the 1000 "
                        "pair_ids already trained, to be subtracted).")
    p.add_argument("--round4-drop-log-json", required=True,
                   help="Path to round-4 drop_log.json (image_path_disk_missing "
                        "543 pair_ids reused as round-5 disk filter).")
    p.add_argument("--recipe-id", required=True,
                   help="Frozen recipe_id pin (sha256[:16] of canonical recipe "
                        "YAML). Round-5 inherits round-4 recipe_id "
                        "= 6bef6e104cdd3442.")
    p.add_argument("--out-dir", required=True,
                   help="Output dir (a UTC subdir is created).")
    p.add_argument("--seed-namespace", default="round5-warm-tier_b-1202-cond-present",
                   help="Seed namespace string for deterministic shuffle of the "
                        "official 1202-pair train set.")
    p.add_argument("--audit-seed-namespace", default="round5-warm-tier_b-1745-setminus-audit",
                   help="Seed namespace string for the audit-trail 1745-pair "
                        "shuffle (independent from the official 1202 shuffle).")
    return p.parse_args(argv)


def compute_seed(namespace: str, recipe_id: str) -> str:
    payload = f"{namespace}||{recipe_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def canonical_pair_ids_sha256(pair_ids: list[str]) -> str:
    """Newline-joined-with-trailing-newline canonical hash.

    Mirrors round-4 ``build_round4_tier_b_1k.canonical_pair_ids_sha256`` so
    trainer pin verification stays a single algorithm across rounds.
    """
    payload = ("\n".join(pair_ids) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shuffled_with_seed(pair_ids: list[str], namespace: str, recipe_id: str):
    """Round-4 protocol: deterministic shuffle, returns (shuffled_list, seed_hex)."""
    seed_hex = compute_seed(namespace, recipe_id)
    rng = random.Random(int(seed_hex, 16))
    out = list(pair_ids)
    rng.shuffle(out)
    return out, seed_hex


def main(argv=None):
    args = parse_args(argv)

    t3_path = pathlib.Path(args.t3_subset_json)
    raw_t3_bytes = t3_path.read_bytes()
    t3 = json.loads(raw_t3_bytes)
    t3_subset_sha256_full = hashlib.sha256(raw_t3_bytes).hexdigest()
    raw_2745 = list(t3["tier_b"]["pair_ids"])
    if len(raw_2745) != 2745:
        sys.stderr.write(
            f"WARN: tier_b.pair_ids count={len(raw_2745)} (expected 2745); "
            f"continuing but this is unusual.\n"
        )

    r4_path = pathlib.Path(args.round4_subset_json)
    r4 = json.loads(r4_path.read_bytes())
    round4_used_1000 = list(r4["tier_b_round4_1k"]["pair_ids"])

    drop_path = pathlib.Path(args.round4_drop_log_json)
    drop_log = json.loads(drop_path.read_bytes())
    disk_missing_records = drop_log["details"]["image_path_disk_missing"]
    disk_missing_pair_ids = [d["pair_id"] for d in disk_missing_records]

    raw_set = set(raw_2745)
    used_set = set(round4_used_1000)
    miss_set = set(disk_missing_pair_ids)

    if not used_set.issubset(raw_set):
        sys.stderr.write(
            f"ERROR: round-4 used pair_ids not subset of raw 2745: "
            f"{len(used_set - raw_set)} stray ids.\n"
        )
        sys.exit(2)
    if not miss_set.issubset(raw_set):
        sys.stderr.write(
            f"ERROR: disk-missing pair_ids not subset of raw 2745: "
            f"{len(miss_set - raw_set)} stray ids.\n"
        )
        sys.exit(3)
    if used_set & miss_set:
        sys.stderr.write(
            f"ERROR: round-4 used overlaps disk-missing "
            f"({len(used_set & miss_set)}); contradicts round-4 task #19 filter.\n"
        )
        sys.exit(4)

    # Two views.
    cond_present_set = raw_set - miss_set            # 2202
    full_setminus_set = raw_set - used_set           # 1745 (audit)
    effective_set = cond_present_set - used_set      # 1202 (official train)

    # Shuffle preserves round-4 protocol; seeds derived from independent
    # namespaces so the official and audit lists are not byte-identical
    # prefixes.
    full_shuffled, full_seed_hex = shuffled_with_seed(
        sorted(full_setminus_set), args.audit_seed_namespace, args.recipe_id,
    )
    effective_shuffled, effective_seed_hex = shuffled_with_seed(
        sorted(effective_set), args.seed_namespace, args.recipe_id,
    )

    full_sha256_full = canonical_pair_ids_sha256(full_shuffled)
    effective_sha256_full = canonical_pair_ids_sha256(effective_shuffled)
    full_sha256_hex16 = full_sha256_full[:16]
    effective_sha256_hex16 = effective_sha256_full[:16]

    # Sanity: effective sha must differ from round-4 sha.
    round4_sha_pin = (
        pathlib.Path(args.round4_subset_json).parent / "pair_ids_sha256_hex16_pin"
    )
    if round4_sha_pin.exists():
        round4_sha = round4_sha_pin.read_text().strip()
        if effective_sha256_hex16 == round4_sha:
            sys.stderr.write(
                f"ERROR: effective sha collides with round-4 sha "
                f"{round4_sha}; this is impossible if setminus is non-empty.\n"
            )
            sys.exit(5)

    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.out_dir) / utc
    out_dir.mkdir(parents=True, exist_ok=True)

    common_meta = {
        "task": "round5-warm-start-setminus",
        "tier_b_source_subset": str(t3_path),
        "tier_b_source_subset_sha256": t3_subset_sha256_full,
        "tier_b_source_pair_count": len(raw_2745),
        "round4_subset_source": str(r4_path),
        "round4_used_pair_count": len(round4_used_1000),
        "round4_used_pair_ids_sha256_hex16": (
            round4_sha_pin.read_text().strip()
            if round4_sha_pin.exists() else None
        ),
        "disk_missing_source": str(drop_path),
        "disk_missing_pair_count": len(disk_missing_pair_ids),
        "filter": {
            "cond_image_present_pair_count": len(cond_present_set),
            "method": (
                "raw 2745 setminus disk_missing 543; mirrored from round-4 "
                "task #19 build (filter snapshot frozen)."
            ),
            "drop_reason_counts": drop_log.get("counts", {}),
        },
        "selection": {
            "official_cardinality": len(effective_set),
            "audit_cardinality": len(full_setminus_set),
            "official_seed_namespace": args.seed_namespace,
            "official_seed_hex8": effective_seed_hex,
            "audit_seed_namespace": args.audit_seed_namespace,
            "audit_seed_hex8": full_seed_hex,
            "recipe_id_pin": args.recipe_id,
            "method": (
                "sorted(setminus) -> sha256(namespace||recipe_id)[:8] -> "
                "Random.shuffle; canonical sha = newline-joined-with-"
                "trailing-newline of the shuffled list."
            ),
        },
        "pair_ids_sha256_canonical": {
            "form": "newline-joined-with-trailing-newline",
            "code": (
                "hashlib.sha256(('\\n'.join(pair_ids) + '\\n')."
                "encode('utf-8')).hexdigest()"
            ),
            "official_sha256_full": effective_sha256_full,
            "official_sha256_hex16": effective_sha256_hex16,
            "audit_sha256_full": full_sha256_full,
            "audit_sha256_hex16": full_sha256_hex16,
        },
        "utc": utc,
    }

    official_payload = {
        "meta": dict(common_meta, role="official-train-set"),
        "tier_b_round5_warm_1202": {
            "pair_ids": effective_shuffled,
            "cardinality": len(effective_shuffled),
            "label": "official-train-set-1202-cond-image-present",
            "sha256_hex16": effective_sha256_hex16,
        },
    }
    audit_payload = {
        "meta": dict(common_meta, role="audit-trail"),
        "tier_b_round5_audit_1745": {
            "audit_pair_ids": full_shuffled,
            "cardinality": len(full_shuffled),
            "label": (
                "audit-trail-1745-raw-setminus-incl-disk-missing; field name "
                "is 'audit_pair_ids' (not 'pair_ids') so a trainer "
                "accidentally pointed at this file still cannot pick it up "
                "via the unique-top-level-pair_ids walk."
            ),
            "sha256_hex16": full_sha256_hex16,
        },
    }

    official_path = out_dir / "T3_round5_warm_official_1202.json"
    audit_path = out_dir / "T3_round5_warm_audit_1745.json"
    official_path.write_bytes(
        (json.dumps(official_payload, ensure_ascii=False,
                    indent=2, sort_keys=False) + "\n").encode("utf-8")
    )
    audit_path.write_bytes(
        (json.dumps(audit_payload, ensure_ascii=False,
                    indent=2, sort_keys=False) + "\n").encode("utf-8")
    )

    (out_dir / "recipe_id_pin").write_text(args.recipe_id + "\n", encoding="utf-8")
    (out_dir / "pair_ids_cond_image_present_sha256_hex16_pin").write_text(
        effective_sha256_hex16 + "\n", encoding="utf-8",
    )
    (out_dir / "pair_ids_full_setminus_sha256_hex16_pin").write_text(
        full_sha256_hex16 + "\n", encoding="utf-8",
    )

    drop_path_out = out_dir / "drop_log.json"
    drop_payload = {
        "counts": drop_log.get("counts", {}),
        "note": (
            "Mirrored from round-4 build/drop_log.json. The 543 "
            "image_path_disk_missing pair_ids are excluded from round-5 "
            "official train set (1202). They remain in the audit-trail 1745 "
            "list."
        ),
        "details": drop_log["details"],
    }
    drop_path_out.write_bytes(
        (json.dumps(drop_payload, ensure_ascii=False, indent=2) + "\n")
        .encode("utf-8")
    )

    print(
        f"OK round-5 warm-start sets built in {out_dir}\n"
        f"  raw_2745={len(raw_2745)} round4_used={len(round4_used_1000)} "
        f"disk_missing={len(disk_missing_pair_ids)}\n"
        f"  cond_present={len(cond_present_set)} "
        f"effective(official)={len(effective_set)} "
        f"full(audit)={len(full_setminus_set)}\n"
        f"  official file: {official_path}\n"
        f"  audit    file: {audit_path}\n"
        f"  effective sha[:16]={effective_sha256_hex16} "
        f"(trainer-asserted)\n"
        f"  audit     sha[:16]={full_sha256_hex16} (provenance only)\n"
        f"  recipe_id={args.recipe_id}"
    )


if __name__ == "__main__":
    main()
