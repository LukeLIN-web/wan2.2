#!/usr/bin/env python3
"""Pick 3 new A-class prompts for eval-v2 from the unused pool."""

import argparse
import hashlib
import json
import pathlib
import random
import re
import sys


SUBTOPIC_RULES = [
    ("racquet_or_paddle", re.compile(r"\b(racquet|paddle|volley|ping[- ]pong|tennis (ball )?(racket|stroke))\b", re.I)),
    ("bowling_pins",      re.compile(r"\bbowling|pins\b", re.I)),
    ("body_vs_body",      re.compile(r"\b(skater|surfer|player|hurdler|athlete)s?\b.*\b(collide|crash|hit|struck|bumped)\b", re.I)),
    ("ball_into_wall",    re.compile(r"\b(ball|softball|kickball)\b.*\b(wall|pane|board|plate)\b", re.I)),
    ("hammer_bounce",     re.compile(r"\bhammer\b.*\b(bounc|drop)", re.I)),
    ("newton_cradle",     re.compile(r"newton'?s? cradle", re.I)),
]

A_KEYWORDS = re.compile(
    r"\b("
    r"collid\w*|collision|collisions|"
    r"impact\w*|"
    r"bounc\w+|rebound\w*|ricochet\w*|"
    r"hits?|strikes?|struck|smash\w*|crash\w*|"
    r"knock(?:s|ing)?\s+(?:into|down|over)|"
    r"rolls? (?:into|toward|against)|"
    r"strike\w*"
    r")\b",
    re.I,
)

NEGATIVE_DOMAIN = [
    ("E_chain",   re.compile(r"\bdomino(es)?\b|\bcd stack\b|\brotating (?:rod|stick) sweep", re.I)),
    ("C_fluid",   re.compile(r"\b(water|liquid|fluid|pour|splash|boil|melt|steam|smoke|liquid)\b", re.I)),
    ("D_optical", re.compile(r"\b(shadow|reflection|mirror|spotlight|illuminated|projection screen|silhouette)\b", re.I)),
    ("B_destroy", re.compile(r"\b(burn\w*|extinguish\w*|tear|rip|fracture|shatter\w*)\b", re.I)),
    ("G_traj",    re.compile(r"\b(parabolic|frisbee|discus|pole vault|sandpit|jumps?\s+(forward|over|onto)|grasshopper)\b", re.I)),
]


def pid12(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def covered_subtopics(captions):
    found = set()
    for cap in captions:
        for name, regex in SUBTOPIC_RULES:
            if regex.search(cap):
                found.add(name)
    return found


def classify_pool_entry(prompt: str, existing_subtopics: set):
    a_match = bool(A_KEYWORDS.search(prompt))
    neg_hits = [name for name, rx in NEGATIVE_DOMAIN if rx.search(prompt)]
    sub_overlap = []
    for name, regex in SUBTOPIC_RULES:
        if regex.search(prompt) and name in existing_subtopics:
            sub_overlap.append(name)
    novel_subtopic = a_match and not sub_overlap
    if not a_match:
        tier = "non_A"
    elif neg_hits and not novel_subtopic:
        tier = "borderline_A"
    elif neg_hits:
        tier = "weak_A"
    elif sub_overlap:
        tier = "duplicate_A_subtopic"
    else:
        tier = "strong_A"
    return {
        "tier": tier,
        "a_match": a_match,
        "neg_hits": neg_hits,
        "subtopic_overlap": sub_overlap,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-t3-root", default="/shared/user59/eval_l40s_test/T0_T3_root")
    ap.add_argument("--round4-out", default="humanize/dpo_v0/out/round4/20260428T160839Z/T3_round4_tier_b_1k.json")
    ap.add_argument("--round5-out", default="humanize/dpo_v0/out/round5/20260430T171646Z/T3_round5_warm_official_1202.json")
    ap.add_argument("--evalprompt", default="humanize/dpo_v0/docs/eval/evalprompt.md")
    ap.add_argument("--target-n", type=int, default=3)
    ap.add_argument("--seed-namespace", default="eval-v2-A-pick")
    ap.add_argument("--show-pool", action="store_true", help="Print full pool with tiers")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.t0_t3_root)
    pairs = json.loads((root / "pair.json").read_bytes())
    heldout = json.loads((root / "splits/heldout.json").read_bytes())
    r4 = json.loads(pathlib.Path(args.round4_out).read_bytes())
    r5 = json.loads(pathlib.Path(args.round5_out).read_bytes())

    pair_by_id = {p["pair_id"]: p for p in pairs}
    heldout_prompts = {p["prompt"] for p in heldout}
    r4_prompts = {pair_by_id[pid]["prompt"] for pid in r4["tier_b_round4_1k"]["pair_ids"]}
    r5_prompts = {pair_by_id[pid]["prompt"] for pid in r5["tier_b_round5_warm_1202"]["pair_ids"]}
    used = heldout_prompts | r4_prompts | r5_prompts
    all_prompts = {p["prompt"] for p in pairs}
    pool = sorted(all_prompts - used)

    existing_A_section = re.search(
        r"### A\..*?\n(?:.*?\n)*?(?=### B\.)",
        pathlib.Path(args.evalprompt).read_text(encoding="utf-8"),
    )
    existing_A_captions = []
    if existing_A_section:
        for line in existing_A_section.group(0).splitlines():
            m = re.match(r"\|\s*`([0-9a-f]{12})`\s*\|\s*([^|]+?)\s*\|", line)
            if m:
                existing_A_captions.append((m.group(1), m.group(2)))
    existing_subtopics = covered_subtopics([c for _, c in existing_A_captions])

    classified = []
    for prompt in pool:
        info = classify_pool_entry(prompt, existing_subtopics)
        classified.append({"pid": pid12(prompt), "prompt": prompt, **info})

    by_tier = {}
    for c in classified:
        by_tier.setdefault(c["tier"], []).append(c)

    seed_bytes = hashlib.sha256(args.seed_namespace.encode()).digest()[:8]
    seed_int = int.from_bytes(seed_bytes, "big")
    rng = random.Random(seed_int)

    rank_order = ["strong_A", "weak_A", "duplicate_A_subtopic", "borderline_A"]
    picks = []
    for tier in rank_order:
        bucket = sorted(by_tier.get(tier, []), key=lambda c: c["pid"])
        rng.shuffle(bucket)
        for c in bucket:
            if len(picks) >= args.target_n:
                break
            picks.append(c)
        if len(picks) >= args.target_n:
            break

    out = {
        "meta": {
            "seed_namespace": args.seed_namespace,
            "seed_int": seed_int,
            "pool_size": len(pool),
            "target_n": args.target_n,
            "tier_counts": {t: len(by_tier.get(t, [])) for t in rank_order + ["non_A"]},
            "existing_A_subtopics": sorted(existing_subtopics),
            "all_existing_A_pids": [p for p, _ in existing_A_captions],
        },
        "picks": [
            {"pid": p["pid"], "tier": p["tier"], "subtopic_overlap": p["subtopic_overlap"],
             "neg_hits": p["neg_hits"], "prompt": p["prompt"]}
            for p in picks
        ],
    }
    if args.show_pool:
        out["pool_classification"] = [
            {"pid": c["pid"], "tier": c["tier"], "neg_hits": c["neg_hits"],
             "subtopic_overlap": c["subtopic_overlap"], "prompt_head": c["prompt"][:140]}
            for c in classified
        ]
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
