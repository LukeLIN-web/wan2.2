#!/usr/bin/env python3
"""Round-6 task #50 sub-task A.1: build the v2-aligned 800-pair tier_b subset.

Per ``humanize/dpo_v0/docs/exp-plan/round6_plan.md`` (frozen plan), round-6
trains on a class-rebalanced 800-pair budget drawn from the cond-present
2202 pool. Per-class quotas are derived from the eval-v2 distribution
(``humanize/dpo_v0/eval/PROMPT_CLASS.json`` v2, n=43):

    A 多体碰撞    279 (≈34.9%)
    B 破坏/形变  167 (≈20.9%)
    C 流体        112 (≈14.0%)
    D 阴影/反射    93 (≈11.6%)
    E 链式          19 (≈ 2.3%)
    F 滚动/滑动    74 (≈ 9.3%)
    G 抛掷/弹道    56 (≈ 7.0%)
    --------------------------
    TOTAL         800

Inputs (frozen path provenance, cited in run_manifest):

- ``--t3-subset-json``  round-2 raw 2745 pool (`tier_b.pair_ids`).
- ``--round4-drop-log-json``  the 543 ``image_path_disk_missing`` pair_ids
  to subtract → 2202 cond-present pool.
- ``--pair-json``  per-pair metadata (``prompt``, ``physical_laws``).
- ``--prompt-class-json``  v2 PROMPT_CLASS oracle for the 43 heldout pids
  (used as ground truth for the classifier round-trip assertion).
- ``--recipe-id``  recipe_id pin (carried into the seed for deterministic
  shuffle, mirroring round-4/round-5 build scripts).

Outputs (gitignored, like round-5):

- ``out/round6/<UTC>/T3_round6_v2aligned_800.json``  the canonical 800-pair
  manifest, with per-class realized n, seed namespace, and pair_ids list
  shuffled deterministically.
- ``out/round6/<UTC>/pair_ids_round6_v2aligned_800_sha256_hex16_pin``  the
  newline-canonical sha256[:16] pin (the value the trainer asserts).
- ``out/round6/<UTC>/recipe_id_pin``  echoed recipe_id pin.
- ``out/round6/<UTC>/class_oracle_v2_verify.json``  the 43-pid round-trip
  proof (assert 43/43 match against PROMPT_CLASS.json v2 BEFORE sampling).
- ``out/round6/<UTC>/per_class_breakdown.json``  per-class pool size,
  realized n, target n, downsample ratio.

The classifier rule is a verbatim port of
``humanize/dpo_v0/docs/data/distirbution.md`` L86-155 (verified 42/42 on
heldout v1; this script re-verifies 43/43 against v2 BEFORE sampling and
halts if the v2 oracle disagrees with the rule, per round6_plan.md
decision 2).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any


CANONICAL_QUOTAS: dict[str, int] = {
    "A": 279,
    "B": 167,
    "C": 112,
    "D": 93,
    "E": 19,
    "F": 74,
    "G": 56,
}
CANONICAL_TOTAL: int = 800
DEFAULT_SEED_NAMESPACE: str = "round6-v2aligned-tier_b-800-cond-present"


# ---------------------------------------------------------------------------
# Class oracle (verbatim port of distirbution.md L86-155).
# ---------------------------------------------------------------------------
def classify(text: str, laws: list[str] | set[str]) -> str:
    """Returns one of A-G or 'unclassified'.

    Verbatim port of ``humanize/dpo_v0/docs/data/distirbution.md`` L86-155.
    Verified 42/42 on heldout v1; this round-6 sampler re-verifies 43/43 on
    v2 in main() before any sampling step.
    """
    t = text.lower()
    laws = set(laws)

    # E - chain / multi-stage
    if "domino" in t or "cascade" in t or "chain reaction" in t \
       or re.search(r"stack of (cd|cards|book)", t) \
       or "stack loses its balance and collapses" in t:
        return "E"

    # B - destruction (verb-driven, runs before A/F)
    # Note: bare "scatter" was removed from this list 2026-04-30 — it
    # over-matched collision-rebound prompts like "collides with the
    # stationary balls, causing them to scatter" (v2 pid 5b7bb71f101d, an
    # A-class addition per eval_v2_changeset.json). Re-verified 41/41
    # on v1 PROMPT_CLASS and 43/43 on v2 PROMPT_CLASS post-removal.
    destr = ["shatter", "breaks into", "breaks open", "break open", "breaks under",
             "breaks,", "broke,", "breaking into", "breaking open", "splits",
             "split open", "bursts", "smash", "crush", "snap", "splinter",
             "cut down", "fell a small tree", "felled", "breaking some",
             "fall to the ground", "falls to the ground", "falling to the ground",
             "leaving water residue", "shattering",
             "burnt", "burns", "burn ", "turns into ash",
             "pops and deflates", "pops and ", "bubble pops", "popped"]
    if any(k in t for k in destr):
        return "B"

    # D - shadow/reflection (require optic word, exclude destructive)
    if ("shadow" in laws) or ("reflection" in laws):
        if any(k in t for k in ["shadow", "reflect", "mirror", "reflections",
                                "glistening", "glints", "light shifts"]) \
           and not any(k in t for k in ["shatter", "break", "crack", "smash", "crush"]):
            return "D"

    # A - explicit collision
    explicit = any(k in t for k in [
        "collid", "hits the", "strikes", "striking ", "rebound",
        "bounces", "bouncing", " bounce", "crashes", "crash", " meet,", "meets,",
        "against a brick wall", "against the wall", "into the corner",
        "off two walls", "pin", "dent", "newton's cradle", "reach for a volley",
        "racquets meet", "hits the gate", "causing it to collapse", "popping out"])
    if explicit and ("collision" in laws or "impenetrability" in laws):
        return "A"

    # C - fluids
    if (laws & {"flow_dynamics", "fluid_continuity", "displacement", "buoyancy"}) \
       or any(k in t for k in ["water", "liquid", "pour", "splash", "fluid",
                              "toothpaste", "boil", "melt", "submerge", "wading"]):
        return "C"

    # G - throw / projectile (no collision target focus)
    if any(k in t for k in ["throws a hammer", "dart lands", "thrown and caught",
                            "hurl", "projectile",
                            "javelin's flight", "releasing the javelin",
                            "throws a softball", "throws a ",
                            "spins noticeably as it travels",
                            "travels through the air"]):
        return "G"

    # F - rolling / sliding / continuous-motion
    slide = [" rolls ", "rolling", "rolled", "glides", "gliding", "slides", "sliding",
             "skate", " drives ", "driving", "swing", "swings", "momentum from",
             "coast", "propelled", "pushes off", "begins to roll", "grounder",
             "shakes its head", "dislodging"]
    if any(k in t for k in slide):
        if any(k in t for k in ["hitting the", "hits the", "strikes the"]):
            return "A"
        return "F"

    # A fallback: collision law + weak verb
    if "collision" in laws and any(k in t for k in
       ["hit", "strike", "impact", "meets", "meet"]):
        return "A"

    # F gravity+inertia falling fallback
    if {"gravity", "inertia"} <= laws and any(k in t for k in
       ["fall", "drops", "dropped", "falls", "falling", "lifts", "lifted",
        "comes loose", "tumbles"]):
        return "F"

    if "collision" in laws:
        return "A"
    return "unclassified"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--t3-subset-json", required=True,
                   help="Path to round-2 T3_subset.json (raw 2745 source).")
    p.add_argument("--round4-drop-log-json", required=True,
                   help="Path to round-4 drop_log.json "
                        "(image_path_disk_missing 543 pair_ids → setminus).")
    p.add_argument("--pair-json", required=True,
                   help="Path to pair.json (pair_id→{prompt, physical_laws, ...}).")
    p.add_argument("--prompt-class-json", required=True,
                   help="Path to humanize/dpo_v0/eval/PROMPT_CLASS.json v2 "
                        "(43 heldout pids, used for classifier round-trip "
                        "verification per round6_plan.md decision 2).")
    p.add_argument("--recipe-id", required=True,
                   help="Frozen recipe_id pin (sha256[:16] of canonical "
                        "recipe YAML); round-6 inherits round-4/round-5's "
                        "6bef6e104cdd3442.")
    p.add_argument("--out-dir", required=True,
                   help="Output dir; a UTC subdir is created.")
    p.add_argument("--seed-namespace", default=DEFAULT_SEED_NAMESPACE,
                   help="Seed namespace string for deterministic shuffle.")
    return p.parse_args(argv)


def compute_seed(namespace: str, recipe_id: str, salt: str = "") -> str:
    payload = f"{namespace}||{recipe_id}||{salt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def canonical_pair_ids_sha256(pair_ids: list[str]) -> str:
    """Newline-joined-with-trailing-newline canonical hash. Same algorithm
    as round-4 ``build_round4_tier_b_1k.canonical_pair_ids_sha256`` and
    round-5 ``build_round5_warm_setminus.canonical_pair_ids_sha256``."""
    payload = ("\n".join(pair_ids) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shuffle_with_seed(items: list[str], namespace: str,
                      recipe_id: str, salt: str = "") -> tuple[list[str], str]:
    """Round-4/5 protocol: deterministic shuffle, returns
    (shuffled_list, seed_hex)."""
    seed_hex = compute_seed(namespace, recipe_id, salt)
    rng = random.Random(int(seed_hex, 16))
    out = list(items)
    rng.shuffle(out)
    return out, seed_hex


def prompt_id_of(prompt: str) -> str:
    """``prompt_id = sha256(prompt)[:12]`` — same canonical form used in
    PROMPT_CLASS.json and evalprompt.md."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ----- Load inputs -----
    t3 = json.loads(pathlib.Path(args.t3_subset_json).read_bytes())
    raw_2745 = list(t3["tier_b"]["pair_ids"])

    drop_log = json.loads(pathlib.Path(args.round4_drop_log_json).read_bytes())
    disk_missing_pair_ids = {
        d["pair_id"] for d in drop_log["details"]["image_path_disk_missing"]
    }

    pair_records: list[dict[str, Any]] = json.loads(
        pathlib.Path(args.pair_json).read_bytes()
    )
    # ``pair.json`` stores ``physical_laws`` as a JSON-encoded *string*
    # (not a list). Decode in-place so the classifier sees a real list/set.
    for r in pair_records:
        laws = r.get("physical_laws")
        if isinstance(laws, str):
            try:
                r["physical_laws"] = json.loads(laws)
            except json.JSONDecodeError:
                # Fallback: leave as-is so the assertion below catches it.
                pass
    pair_by_id: dict[str, dict[str, Any]] = {r["pair_id"]: r for r in pair_records}

    prompt_class_v2 = json.loads(pathlib.Path(args.prompt_class_json).read_bytes())
    if prompt_class_v2.get("version") != "2":
        sys.stderr.write(
            f"WARN: PROMPT_CLASS.json version={prompt_class_v2.get('version')!r} "
            "(expected '2'); continuing but flag this in commit msg.\n"
        )
    v2_pid_to_class: dict[str, str] = dict(prompt_class_v2["prompts"])

    # ----- Step 0: classifier oracle re-verify on v2 (43/43 assert) -----
    # We classify each v2 heldout prompt using the rule (which needs the
    # source prompt text + physical_laws); we recover those by looking up
    # ANY pair in pair_records whose prompt has the matching prompt_id (all
    # such pairs share prompt + laws since prompt is the grouping key).
    v2_pid_to_prompt_record: dict[str, dict[str, Any]] = {}
    for r in pair_records:
        pid = prompt_id_of(r["prompt"])
        if pid in v2_pid_to_class and pid not in v2_pid_to_prompt_record:
            v2_pid_to_prompt_record[pid] = r

    missing_v2_pids = sorted(set(v2_pid_to_class) - set(v2_pid_to_prompt_record))
    if missing_v2_pids:
        sys.stderr.write(
            f"ERROR: {len(missing_v2_pids)} v2 pids have no matching "
            f"prompt in pair.json; cannot run oracle. Missing: "
            f"{missing_v2_pids[:5]}\n"
        )
        return 2

    oracle_disagreements: list[dict[str, Any]] = []
    for pid, expected in v2_pid_to_class.items():
        rec = v2_pid_to_prompt_record[pid]
        actual = classify(rec["prompt"], rec.get("physical_laws", []))
        if actual != expected:
            oracle_disagreements.append({
                "prompt_id": pid,
                "expected": expected,
                "actual": actual,
                "prompt": rec["prompt"],
            })
    if oracle_disagreements:
        sys.stderr.write(
            f"ERROR: classifier oracle disagrees with PROMPT_CLASS.json v2 "
            f"on {len(oracle_disagreements)} of {len(v2_pid_to_class)} "
            f"prompt(s). First: "
            f"{json.dumps(oracle_disagreements[0], ensure_ascii=False)[:300]}.\n"
            "Per round6_plan.md decision 2, sampler is BLOCKED on this "
            "assertion. Either fix the rule or update PROMPT_CLASS.json.\n"
        )
        return 3
    print(f"[oracle] v2 round-trip OK: 43/43 match "
          f"({len(v2_pid_to_class)} prompts).")

    # ----- Build cond-present 2202 pool -----
    raw_set = set(raw_2745)
    cond_present_set = raw_set - disk_missing_pair_ids
    if len(cond_present_set) != 2202:
        sys.stderr.write(
            f"WARN: cond_present pool size {len(cond_present_set)} != 2202 "
            f"(round-5 baseline). Continuing with sampled count.\n"
        )

    # ----- Bucket cond-present pair_ids by class -----
    pool_by_class: dict[str, list[str]] = defaultdict(list)
    pool_unclassified: list[str] = []
    pool_missing_pair_record: list[str] = []
    for pid in sorted(cond_present_set):
        rec = pair_by_id.get(pid)
        if rec is None:
            pool_missing_pair_record.append(pid)
            continue
        cls = classify(rec["prompt"], rec.get("physical_laws", []))
        if cls == "unclassified":
            pool_unclassified.append(pid)
        else:
            pool_by_class[cls].append(pid)

    pool_summary = {
        cls: len(pool_by_class[cls]) for cls in sorted(pool_by_class)
    }
    pool_summary["unclassified"] = len(pool_unclassified)
    if pool_missing_pair_record:
        pool_summary["pair_record_missing"] = len(pool_missing_pair_record)
    print(f"[pool] cond-present 2202 bucketed: {pool_summary}")

    # ----- Halt if any class can't meet its quota -----
    short_classes: list[tuple[str, int, int]] = []
    for cls, target in CANONICAL_QUOTAS.items():
        have = len(pool_by_class.get(cls, []))
        if have < target:
            short_classes.append((cls, have, target))
    if short_classes:
        sys.stderr.write(
            f"ERROR: per-class quotas cannot be met from cond-present 2202 "
            f"pool: {short_classes}. Per round6_plan.md decision 1 "
            f"(zero-buffer freeze on B), this is a halt condition.\n"
        )
        return 4

    # ----- Per-class deterministic shuffle + take quota -----
    selected_by_class: dict[str, list[str]] = {}
    seed_hexes_by_class: dict[str, str] = {}
    for cls, target in CANONICAL_QUOTAS.items():
        # Salt with the class letter so different classes get independent
        # shuffles even though they share the same namespace.
        shuffled, seed_hex = shuffle_with_seed(
            pool_by_class[cls],
            args.seed_namespace,
            args.recipe_id,
            salt=cls,
        )
        selected_by_class[cls] = shuffled[:target]
        seed_hexes_by_class[cls] = seed_hex

    # Concatenate in alphabetical class order, then deterministically
    # shuffle the full 800 list once more so the trainer's
    # DistributedSampler (shuffle=False) sees an interleaved stream rather
    # than a class-block-sorted one (round-5 protocol consistency).
    concat = []
    for cls in sorted(selected_by_class):
        concat.extend(selected_by_class[cls])
    if len(concat) != CANONICAL_TOTAL:
        sys.stderr.write(
            f"ERROR: concat size {len(concat)} != {CANONICAL_TOTAL}\n"
        )
        return 5
    selected_shuffled, full_seed_hex = shuffle_with_seed(
        concat,
        args.seed_namespace,
        args.recipe_id,
        salt="full-interleave",
    )
    pair_ids_sha256_full = canonical_pair_ids_sha256(selected_shuffled)
    pair_ids_sha256_hex16 = pair_ids_sha256_full[:16]

    realized_per_class = {
        cls: len(selected_by_class[cls]) for cls in CANONICAL_QUOTAS
    }
    if realized_per_class != CANONICAL_QUOTAS:
        sys.stderr.write(
            f"ERROR: realized per-class counts {realized_per_class} != "
            f"target {CANONICAL_QUOTAS}\n"
        )
        return 6

    # ----- Write outputs -----
    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.out_dir) / utc
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "task": "round6-class-balanced-sampler",
            "tier_b_source_subset": str(args.t3_subset_json),
            "tier_b_source_pair_count": len(raw_2745),
            "round4_drop_log_source": str(args.round4_drop_log_json),
            "disk_missing_pair_count": len(disk_missing_pair_ids),
            "cond_present_pair_count": len(cond_present_set),
            "pair_json_source": str(args.pair_json),
            "prompt_class_json_source": str(args.prompt_class_json),
            "prompt_class_version": prompt_class_v2.get("version"),
            "v2_oracle_round_trip": {
                "n_pids": len(v2_pid_to_class),
                "n_match": len(v2_pid_to_class) - len(oracle_disagreements),
                "n_disagreement": len(oracle_disagreements),
                "method": "classify(rec.prompt, rec.physical_laws) == "
                          "PROMPT_CLASS.json v2 mapping for all 43 pids",
            },
            "pool": pool_summary,
            "selection": {
                "target_quotas": CANONICAL_QUOTAS,
                "realized_per_class": realized_per_class,
                "target_total": CANONICAL_TOTAL,
                "realized_total": len(selected_shuffled),
                "seed_namespace": args.seed_namespace,
                "per_class_seed_hex": seed_hexes_by_class,
                "full_interleave_seed_hex": full_seed_hex,
                "recipe_id_pin": args.recipe_id,
                "method": (
                    "for each class: sorted(pool[class]) -> "
                    "sha256(namespace||recipe_id||class)[:8] -> "
                    "Random.shuffle -> take quota; concat in alphabetical "
                    "class order; full sha256(namespace||recipe_id||"
                    "'full-interleave')[:8] -> Random.shuffle once more; "
                    "canonical sha = newline-joined-with-trailing-newline of "
                    "the final shuffled list."
                ),
            },
            "pair_ids_sha256_canonical": {
                "form": "newline-joined-with-trailing-newline",
                "code": (
                    "hashlib.sha256(('\\n'.join(pair_ids) + '\\n')."
                    "encode('utf-8')).hexdigest()"
                ),
                "sha256_full": pair_ids_sha256_full,
                "sha256_hex16": pair_ids_sha256_hex16,
            },
            "utc": utc,
        },
        "tier_b_round6_v2aligned_800": {
            "pair_ids": selected_shuffled,
            "cardinality": len(selected_shuffled),
            "label": "official-train-set-800-v2-class-balanced",
            "sha256_hex16": pair_ids_sha256_hex16,
        },
    }

    out_path = out_dir / "T3_round6_v2aligned_800.json"
    out_path.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
         + "\n").encode("utf-8")
    )

    (out_dir / "recipe_id_pin").write_text(
        args.recipe_id + "\n", encoding="utf-8",
    )
    (out_dir / "pair_ids_round6_v2aligned_800_sha256_hex16_pin").write_text(
        pair_ids_sha256_hex16 + "\n", encoding="utf-8",
    )

    # F-checklist artifact: explicit oracle round-trip proof.
    (out_dir / "class_oracle_v2_verify.json").write_bytes(
        (json.dumps({
            "n_pids": len(v2_pid_to_class),
            "n_match": len(v2_pid_to_class) - len(oracle_disagreements),
            "disagreements": oracle_disagreements,
            "rule_source":
                "humanize/dpo_v0/docs/data/distirbution.md L86-155 (verbatim)",
            "oracle_source":
                "humanize/dpo_v0/eval/PROMPT_CLASS.json v2",
            "utc": utc,
        }, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )

    # F-checklist artifact: per-class breakdown with pool sizes.
    per_class_breakdown = {}
    for cls in sorted(set(CANONICAL_QUOTAS) | set(pool_by_class)):
        pool_n = len(pool_by_class.get(cls, []))
        target_n = CANONICAL_QUOTAS.get(cls, 0)
        realized_n = realized_per_class.get(cls, 0)
        per_class_breakdown[cls] = {
            "pool_n": pool_n,
            "target_n": target_n,
            "realized_n": realized_n,
            "downsample_ratio": (target_n / pool_n) if pool_n else None,
            "buffer": pool_n - target_n,
        }
    per_class_breakdown["unclassified"] = {
        "pool_n": len(pool_unclassified),
        "target_n": 0,
        "realized_n": 0,
        "note": "excluded from training per round6_plan.md (no class slot)",
    }
    (out_dir / "per_class_breakdown.json").write_bytes(
        (json.dumps(per_class_breakdown, ensure_ascii=False,
                    indent=2) + "\n").encode("utf-8")
    )

    print(
        f"OK round-6 v2aligned 800-pair subset built in {out_dir}\n"
        f"  per-class realized: {realized_per_class}\n"
        f"  total: {len(selected_shuffled)}\n"
        f"  pair_ids sha256[:16]: {pair_ids_sha256_hex16}\n"
        f"  oracle v2 round-trip: "
        f"{len(v2_pid_to_class) - len(oracle_disagreements)}/"
        f"{len(v2_pid_to_class)} match\n"
        f"  recipe_id: {args.recipe_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
