"""Post-run multi-rank watchdog summary aggregator.

CLI: ``python -m humanize.dpo_v0.watchdog.aggregate <run_dir>``

Walks ``<run_dir>/watchdog/rank*/summary.json``, merges them into
``<run_dir>/watchdog/summary.json``, and surfaces a top-level cross-rank
report with global VRAM peak, total forwards, total high/low routing
counts, and per-rank loss min/last. Designed to be called once at the
end of a run, after every rank has flushed its own per-rank summary.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def aggregate(run_dir: pathlib.Path) -> dict:
    wd_root = run_dir / "watchdog"
    if not wd_root.exists():
        raise FileNotFoundError(f"no watchdog/ directory under {run_dir}")
    rank_dirs = sorted(p for p in wd_root.iterdir() if p.is_dir() and p.name.startswith("rank"))
    if not rank_dirs:
        raise FileNotFoundError(f"no rank<n>/ subdirs under {wd_root}")
    per_rank: list[dict] = []
    for d in rank_dirs:
        sp = d / "summary.json"
        if not sp.exists():
            continue
        per_rank.append(json.loads(sp.read_text(encoding="utf-8")))

    cross = {
        "n_ranks": len(per_rank),
        "global_peak_alloc_gib": max((r["vram"].get("global_peak_alloc_gib", 0.0) for r in per_rank), default=0.0),
        "global_peak_reserved_gib": max((r["vram"].get("global_peak_reserved_gib", 0.0) for r in per_rank), default=0.0),
        "total_forwards": sum(r["routing"].get("total_forwards", 0) for r in per_rank),
        "total_high_count": sum(r["routing"].get("high_count", 0) for r in per_rank),
        "total_low_count": sum(r["routing"].get("low_count", 0) for r in per_rank),
    }
    cross["total_fraction_high_noise"] = (
        round(cross["total_high_count"] / cross["total_forwards"], 6)
        if cross["total_forwards"] > 0
        else 0.0
    )
    out = {"per_rank": per_rank, "cross_rank": cross}
    (wd_root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=pathlib.Path)
    args = p.parse_args()
    out = aggregate(args.run_dir)
    print(json.dumps(out["cross_rank"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
