#!/usr/bin/env python3
"""Round-7 A 路 RLCR loop sampler: build the v2-aligned 1024-pair tier_b subset.

Per ``humanize/dpo_v0/docs/exp-plan/round7draft.md`` (DRAFT at fork-time),
round-7 trains on a class-rebalanced 1024-pair budget. Round-6 used 800
pairs drawn purely from the cond-present 2202 pool; round-7 enlarges the
budget to 1024 and rescues 42 previously-disk-missing B-class pair_ids
that have been rebuilt on juyi-videorl (cond images + latents verified
non-empty by the upstream pre-flight task).

Per-class quotas (eval-v2 distribution × 1024):

    A 多体碰撞    357 (≈34.9%)
    B 破坏/形变  214 (≈20.9%) = 172 cond-present + 42 rescued
    C 流体        143 (≈14.0%)
    D 阴影/反射   119 (≈11.6%)
    E 链式          24 (≈ 2.3%)
    F 滚动/滑动    95 (≈ 9.3%)
    G 抛掷/弹道    72 (≈ 7.0%)
    --------------------------
    TOTAL        1024

Inputs are identical to round-6 plus a new ``--rescue-b-pair-ids-json``
that carries the 42 rescued B pids. Outputs:
``T3_round7_v2aligned_1024.json``,
``pair_ids_round7_v2aligned_1024_sha256_hex16_pin``,
``recipe_id_pin``, ``class_oracle_v2_verify.json``,
``per_class_breakdown.json``.
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
    "A": 357,
    "B": 214,
    "C": 143,
    "D": 119,
    "E": 24,
    "F": 95,
    "G": 72,
}
CANONICAL_TOTAL: int = 1024
DEFAULT_SEED_NAMESPACE: str = "round7-fresh-tier_b-1024-cond-present"

assert sum(CANONICAL_QUOTAS.values()) == CANONICAL_TOTAL, (
    f"CANONICAL_QUOTAS sum {sum(CANONICAL_QUOTAS.values())} != "
    f"CANONICAL_TOTAL {CANONICAL_TOTAL}"
)


# ---------------------------------------------------------------------------
# Class oracle (verbatim port of distirbution.md L86-155).
# ---------------------------------------------------------------------------
def classify(text: str, laws: list[str] | set[str]) -> str:
    """Returns one of A-G or 'unclassified'.

    Verbatim port of ``humanize/dpo_v0/docs/data/distirbution.md`` L86-155.
    Verified 42/42 on heldout v1; this round-7 sampler re-verifies 43/43 on
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
                        "verification per round7draft.md decision 2).")
    p.add_argument("--recipe-id", required=True,
                   help="Frozen recipe_id pin (sha256[:16] of canonical "
                        "recipe YAML); round-7 inherits round-4/5/6's "
                        "6bef6e104cdd3442.")
    p.add_argument("--b-mode", choices=("rescue", "repeat"), default="repeat",
                   help="How to assemble the 214-pair B class. 'rescue' = "
                        "172 cond-present + 42 disk-missing rebuilt pids "
                        "(plan's primary path; requires --rescue-b-pair-ids-json). "
                        "'repeat' = 172 cond-present + 42 deterministic "
                        "repeats from the same pool (plan §'B rescue + E "
                        "zero-buffer policy' fallback (ii); authorised when "
                        "the disk-missing source images are unrecoverable). "
                        "Default 'repeat' since round-7's source images at "
                        "/shared/.../physics-IQ-benchmark/switch-frames are "
                        "gone. The trainer must be invoked with "
                        "--allow-repeated-pair-ids true in 'repeat' mode.")
    p.add_argument("--rescue-b-pair-ids-json", default=None,
                   help="Required when --b-mode=rescue. Path to a JSON file "
                        "containing exactly 42 disk-missing B-class pair_ids "
                        "that have been rebuilt on juyi-videorl (cond images "
                        "+ latents verified non-empty). Accepts either a "
                        "flat list or a dict of shape "
                        "``{\"rescued_b\": {\"pair_ids\": [...]}}`` "
                        "(see _load_rescue_b_pair_ids). Ignored when "
                        "--b-mode=repeat.")
    p.add_argument("--out-dir", required=True,
                   help="Output dir; a UTC subdir is created.")
    p.add_argument("--seed-namespace", default=DEFAULT_SEED_NAMESPACE,
                   help="Seed namespace string for deterministic shuffle.")
    args = p.parse_args(argv)
    if args.b_mode == "rescue" and args.rescue_b_pair_ids_json is None:
        p.error("--b-mode=rescue requires --rescue-b-pair-ids-json")
    return args


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
    """Round-4/5/6 protocol: deterministic shuffle, returns
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


def _load_rescue_b_pair_ids(path: pathlib.Path) -> list[str]:
    """Load the rescued-B pair_ids list. Accepts two shapes:

    1. flat list:  ``["pid1", "pid2", ...]``
    2. nested:     ``{"rescued_b": {"pair_ids": ["pid1", ...]}}``

    Both are produced by the upstream pre-flight task; we accept both
    rather than forcing one canonical shape.
    """
    obj = json.loads(path.read_bytes())
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        if "rescued_b" in obj and isinstance(obj["rescued_b"], dict) \
                and "pair_ids" in obj["rescued_b"]:
            return list(obj["rescued_b"]["pair_ids"])
    raise ValueError(
        f"--rescue-b-pair-ids-json {path} has unsupported shape; "
        "expected flat list or {'rescued_b': {'pair_ids': [...]}}."
    )


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
            "Per round7draft.md decision 2, sampler is BLOCKED on this "
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

    # ----- Bucket cond-present pair_ids by class (needed for both B modes) -----
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
    print(f"[pool] cond-present {len(cond_present_set)} bucketed: {pool_summary}")

    cond_present_b_count = len(pool_by_class.get("B", []))

    # ----- Resolve the 42 "extra B" pids per --b-mode -----
    # rescued_b_raw is the canonical name for the 42 extras throughout the
    # rest of this script regardless of mode; in 'rescue' mode they are
    # disk-missing-rebuilt, in 'repeat' mode they are deterministic repeats
    # of cond-present-B (causing the trainer to see those 42 pids twice).
    if args.b_mode == "rescue":
        # ----- Load and validate rescued-B pair_ids (4 gates) -----
        rescue_path = pathlib.Path(args.rescue_b_pair_ids_json)
        rescued_b_raw = _load_rescue_b_pair_ids(rescue_path)
        rescued_b_set = set(rescued_b_raw)
        b_repeat_pick_seed = None

        if len(rescued_b_set) != 42 or len(rescued_b_raw) != 42:
            sys.stderr.write(
                f"ERROR: --rescue-b-pair-ids-json must contain exactly 42 "
                f"unique pair_ids; got len={len(rescued_b_raw)} "
                f"unique={len(rescued_b_set)}.\n"
            )
            return 7

        rescued_in_cond_present = rescued_b_set & cond_present_set
        rescued_not_in_disk_missing = rescued_b_set - disk_missing_pair_ids
        if rescued_in_cond_present or rescued_not_in_disk_missing:
            sys.stderr.write(
                f"ERROR: rescued-B pair_ids must all be disk-missing and not "
                f"cond-present. cond_present_overlap="
                f"{sorted(rescued_in_cond_present)[:5]} ({len(rescued_in_cond_present)}), "
                f"not_in_disk_missing="
                f"{sorted(rescued_not_in_disk_missing)[:5]} "
                f"({len(rescued_not_in_disk_missing)}).\n"
            )
            return 8

        rescued_missing_pair_record = [pid for pid in rescued_b_raw if pid not in pair_by_id]
        if rescued_missing_pair_record:
            sys.stderr.write(
                f"ERROR: {len(rescued_missing_pair_record)} rescued-B pair_ids "
                f"have no pair.json record. First 5: "
                f"{rescued_missing_pair_record[:5]}\n"
            )
            return 9

        rescued_class_disagreements: list[dict[str, Any]] = []
        for pid in rescued_b_raw:
            rec = pair_by_id[pid]
            cls = classify(rec["prompt"], rec.get("physical_laws", []))
            if cls != "B":
                rescued_class_disagreements.append({
                    "pair_id": pid,
                    "expected": "B",
                    "actual": cls,
                    "prompt": rec["prompt"],
                })
        if rescued_class_disagreements:
            sys.stderr.write(
                f"ERROR: {len(rescued_class_disagreements)} rescued-B pair_ids "
                f"do not classify as B. First: "
                f"{json.dumps(rescued_class_disagreements[0], ensure_ascii=False)[:300]}\n"
            )
            return 10

        print(f"[rescue-b] gates OK: 42/42 unique, disk-missing-only, "
              f"pair.json-present, classify==B.")
    else:
        # repeat mode: pick 42 pids deterministically from cond_present_B.
        # The 42 will appear *twice* in the final 1024 list (once via the
        # 'B-cond' shuffle that uses all 172 of pool_by_class['B'], once
        # via this 'B-repeat-pick' selection). Trainer must run with
        # --allow-repeated-pair-ids true to honour the multiset.
        cond_b_pool = sorted(pool_by_class.get("B", []))
        if len(cond_b_pool) < 42:
            sys.stderr.write(
                f"ERROR: --b-mode=repeat requires cond_present_B >= 42 "
                f"(have {len(cond_b_pool)}). Cannot pick 42 distinct pids "
                f"to repeat.\n"
            )
            return 7
        repeat_shuffled, b_repeat_pick_seed = shuffle_with_seed(
            cond_b_pool, args.seed_namespace, args.recipe_id,
            salt="B-repeat-pick",
        )
        rescued_b_raw = repeat_shuffled[:42]
        rescued_b_set = set(rescued_b_raw)
        if not rescued_b_set <= set(cond_b_pool):
            sys.stderr.write(
                "ERROR: B-repeat-pick produced pids outside cond_present_B "
                "(invariant violated).\n"
            )
            return 8
        print(f"[b-repeat] picked 42/42 deterministic repeats from "
              f"cond_present_B (pool={len(cond_b_pool)}); "
              f"seed_hex={b_repeat_pick_seed}.")

    # ----- Halt if any class can't meet its unique-pool requirement -----
    # In 'rescue' mode B's pool = cond-present-B + rescued-B. In 'repeat'
    # mode the 42 repeats come from cond-present-B itself, so the pool
    # available for the multiset of 214 is just cond-present-B (must be
    # >= 172 since 42 entries are duplicates).
    short_classes: list[tuple[str, int, int]] = []
    for cls, target in CANONICAL_QUOTAS.items():
        if cls == "B":
            if args.b_mode == "rescue":
                have = cond_present_b_count + len(rescued_b_raw)
                effective_target = target
            else:
                have = cond_present_b_count
                effective_target = target - 42
        else:
            have = len(pool_by_class.get(cls, []))
            effective_target = target
        if have < effective_target:
            short_classes.append((cls, have, effective_target))
    if short_classes:
        sys.stderr.write(
            f"ERROR: per-class pools cannot meet required minimum: "
            f"{short_classes}. Per round7draft.md (zero-buffer freeze on "
            f"B/E), this is a halt condition.\n"
        )
        return 4

    # ----- Per-class deterministic shuffle + take quota -----
    selected_by_class: dict[str, list[str]] = {}
    seed_hexes_by_class: dict[str, str] = {}
    for cls, target in CANONICAL_QUOTAS.items():
        if cls == "B":
            # B = cond-present-B (172) + rescued-B (42) = 214.
            # Shuffle each sub-list deterministically with distinct salts,
            # concat (cond first, then rescued), take all 214 (no truncation).
            cond_b_shuffled, b_cond_seed = shuffle_with_seed(
                pool_by_class["B"],
                args.seed_namespace,
                args.recipe_id,
                salt="B-cond",
            )
            rescue_b_shuffled, b_rescue_seed = shuffle_with_seed(
                rescued_b_raw,
                args.seed_namespace,
                args.recipe_id,
                salt="B-rescue",
            )
            selected_by_class["B"] = cond_b_shuffled + rescue_b_shuffled
            seed_hexes_by_class["B-cond"] = b_cond_seed
            seed_hexes_by_class["B-rescue"] = b_rescue_seed
            if args.b_mode == "repeat" and b_repeat_pick_seed is not None:
                seed_hexes_by_class["B-repeat-pick"] = b_repeat_pick_seed
            if len(selected_by_class["B"]) != target:
                sys.stderr.write(
                    f"ERROR: assembled B size {len(selected_by_class['B'])} "
                    f"!= target {target}\n"
                )
                return 11
            continue
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
    # shuffle the full 1024 list once more so the trainer's
    # DistributedSampler (shuffle=False) sees an interleaved stream rather
    # than a class-block-sorted one (round-5/6 protocol consistency).
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

    rescue_pair_ids_sorted = sorted(rescued_b_set)
    rescue_pair_ids_sha256 = canonical_pair_ids_sha256(rescue_pair_ids_sorted)

    payload = {
        "meta": {
            "task": "round7-class-balanced-sampler",
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
            "b_class_assembly": {
                "b_mode": args.b_mode,
                "rescue_pair_ids_json_source": (
                    str(args.rescue_b_pair_ids_json)
                    if args.b_mode == "rescue" else None
                ),
                "rescue_pair_ids_count": len(rescued_b_raw),
                "disk_missing_pair_count_total": len(disk_missing_pair_ids),
                "cond_present_b_count": cond_present_b_count,
                "rescue_pair_ids_sha256": rescue_pair_ids_sha256,
                "duplicate_pair_count_in_final_list": (
                    0 if args.b_mode == "rescue"
                    else len(rescued_b_raw)
                ),
                "method": (
                    "B = cond_present_B (172) + rescued_B (42) = 214; "
                    "each sub-list deterministically shuffled with distinct "
                    "salts ('B-cond', 'B-rescue'), concatenated cond-first, "
                    "no truncation."
                ) if args.b_mode == "rescue" else (
                    "B = cond_present_B-shuffled (172) + 42 deterministic "
                    "repeats from cond_present_B = 214; salts 'B-cond' "
                    "(full 172 shuffle), 'B-repeat-pick' (42 picks from "
                    "sorted(cond_present_B)), 'B-rescue' (re-shuffle of the "
                    "42 picks before concat). The 42 repeats appear TWICE "
                    "in the final 1024 list — trainer must run with "
                    "--allow-repeated-pair-ids true. This is plan §'B "
                    "rescue + E zero-buffer policy' fallback (ii), "
                    "authorised because the disk-missing source images "
                    "are unrecoverable."
                ),
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
                    "for each class != B: sorted(pool[class]) -> "
                    "sha256(namespace||recipe_id||class)[:8] -> "
                    "Random.shuffle -> take quota; "
                    "B = cond-present + extras, both deterministically "
                    "shuffled (salts 'B-cond', 'B-rescue') and concatenated, "
                    "no truncation; the 42 'extras' are mode-dependent (see "
                    "meta.b_class_assembly); concat in alphabetical class "
                    "order; full sha256(namespace||recipe_id||"
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
        "tier_b_round7_v2aligned_1024": {
            "pair_ids": selected_shuffled,
            "cardinality": len(selected_shuffled),
            "label": (
                "official-train-set-1024-v2-class-balanced-with-b-rescue"
                if args.b_mode == "rescue"
                else "official-train-set-1024-v2-class-balanced-with-b-repeat-fallback"
            ),
            "sha256_hex16": pair_ids_sha256_hex16,
        },
    }

    out_path = out_dir / "T3_round7_v2aligned_1024.json"
    out_path.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
         + "\n").encode("utf-8")
    )

    (out_dir / "recipe_id_pin").write_text(
        args.recipe_id + "\n", encoding="utf-8",
    )
    (out_dir / "pair_ids_round7_v2aligned_1024_sha256_hex16_pin").write_text(
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
    per_class_breakdown: dict[str, dict[str, Any]] = {}
    for cls in sorted(set(CANONICAL_QUOTAS) | set(pool_by_class)):
        target_n = CANONICAL_QUOTAS.get(cls, 0)
        realized_n = realized_per_class.get(cls, 0)
        if cls == "B":
            cond_n = cond_present_b_count
            rescued_n = len(rescued_b_raw)
            if args.b_mode == "rescue":
                pool_n = cond_n + rescued_n
            else:
                pool_n = cond_n
            per_class_breakdown[cls] = {
                "b_mode": args.b_mode,
                "cond_present_n": cond_n,
                "extras_n": rescued_n,
                "pool_n": pool_n,
                "target_n": target_n,
                "realized_n": realized_n,
                "downsample_ratio": (target_n / pool_n) if pool_n else None,
                "buffer": pool_n - target_n,
            }
        else:
            pool_n = len(pool_by_class.get(cls, []))
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
        "note": "excluded from training per round7draft.md (no class slot)",
    }
    (out_dir / "per_class_breakdown.json").write_bytes(
        (json.dumps(per_class_breakdown, ensure_ascii=False,
                    indent=2) + "\n").encode("utf-8")
    )

    print(
        f"OK round-7 v2aligned 1024-pair subset built in {out_dir}\n"
        f"  per-class realized: {realized_per_class}\n"
        f"  total: {len(selected_shuffled)} (unique="
        f"{len(set(selected_shuffled))})\n"
        f"  pair_ids sha256[:16]: {pair_ids_sha256_hex16}\n"
        f"  oracle v2 round-trip: "
        f"{len(v2_pid_to_class) - len(oracle_disagreements)}/"
        f"{len(v2_pid_to_class)} match\n"
        f"  b_class (mode={args.b_mode}): cond={cond_present_b_count} + "
        f"extras={len(rescued_b_raw)} = {len(selected_by_class['B'])}\n"
        f"  recipe_id: {args.recipe_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
