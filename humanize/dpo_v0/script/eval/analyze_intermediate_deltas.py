#!/usr/bin/env python3
"""Compare step50 / step100 / step150 trained PhyJudge scores against the v3
final-ckpt baseline scores, per axis, with paired delta + sign-test p-values.

Reads:
  --baseline-results PATH        v3 full-eval results.jsonl (84 records:
                                 42 baseline + 42 trained). Only baseline rows
                                 are used here.
  --run-root PATH                v3_intermediate_<ts> root produced by
                                 launch_v3_intermediate_regen.sh + scored by
                                 launch_v3_intermediate_score.sh. Per-step
                                 trained results.jsonl is auto-discovered at
                                 <run-root>/scores/step<N>/<ts>/results.jsonl
                                 (newest wins).
  --steps "50 100 150"           which step ids to compare
  --axes "SA PTV persistence inertia momentum"

Mapping: each row's video path is parsed for its prompt_id directory segment
(`heldout_regen/<prompt_id>/{baseline,trained}/...`); rows are paired by
prompt_id. Baseline rows are filtered to mode=baseline by path inspection.

Output: per-step Markdown table with mean Δ, +wins / −losses / =ties counts,
and 2-tailed exact-binomial sign-test p (excluding ties), plus a JSON dump.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROMPT_RE = re.compile(r"/heldout_regen/([0-9a-f]{12})/(baseline|trained)/")


def _exact_binomial_two_tailed(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-tailed p-value for k successes in n trials at H0 prob=p.

    Method: sum the probabilities of all outcomes whose pmf <= pmf(k);
    handles ties at the observed pmf consistently with scipy's binom_test.
    """
    if n == 0:
        return 1.0

    def _pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    obs = _pmf(k)
    eps = 1e-12
    total = 0.0
    for i in range(0, n + 1):
        if _pmf(i) <= obs + eps:
            total += _pmf(i)
    return min(1.0, total)


def _extract_prompt_and_mode(video_or_run_id: str) -> tuple[str, str] | None:
    m = PROMPT_RE.search(video_or_run_id)
    if not m:
        return None
    return m.group(1), m.group(2)


def _row_axis_score(row: dict, axis: str) -> float | None:
    """Pull a numeric score for an axis from a results.jsonl row.

    The scorer shape is one of:
      * row[axis] = {"score": 3, ...}
      * row["scores"] = {axis: {"score": 3, ...} | 3}
      * row[axis] = 3  (fallback)
    """
    cand = row.get(axis)
    if cand is None and isinstance(row.get("scores"), dict):
        cand = row["scores"].get(axis)
    if isinstance(cand, dict):
        cand = cand.get("score")
    try:
        out = float(cand)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _index_by_prompt(
    rows: Iterable[dict],
    want_mode: str | None,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        for key in ("video", "video_path", "run_id"):
            if key in r and isinstance(r[key], str):
                pm = _extract_prompt_and_mode(r[key])
                if pm is not None:
                    pid, mode = pm
                    if want_mode is not None and mode != want_mode:
                        break
                    out[pid] = r
                    break
    return out


def _newest_results_jsonl(scores_dir: Path) -> Path | None:
    if not scores_dir.is_dir():
        return None
    candidates = sorted(scores_dir.glob("*/results.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def compare_step(
    baseline_by_prompt: dict[str, dict],
    trained_by_prompt: dict[str, dict],
    axes: list[str],
) -> dict:
    common = sorted(set(baseline_by_prompt) & set(trained_by_prompt))
    out: dict = {"n_paired": len(common), "axes": {}}
    for axis in axes:
        deltas: list[float] = []
        wins = losses = ties = 0
        b_vals: list[float] = []
        t_vals: list[float] = []
        for pid in common:
            b = _row_axis_score(baseline_by_prompt[pid], axis)
            t = _row_axis_score(trained_by_prompt[pid], axis)
            if b is None or t is None:
                continue
            d = t - b
            deltas.append(d)
            b_vals.append(b)
            t_vals.append(t)
            if d > 0:
                wins += 1
            elif d < 0:
                losses += 1
            else:
                ties += 1
        n_signed = wins + losses
        p = _exact_binomial_two_tailed(min(wins, losses), n_signed) if n_signed else 1.0
        out["axes"][axis] = {
            "n": len(deltas),
            "baseline_mean": (statistics.fmean(b_vals) if b_vals else None),
            "baseline_std": (statistics.pstdev(b_vals) if len(b_vals) > 1 else None),
            "trained_mean": (statistics.fmean(t_vals) if t_vals else None),
            "trained_std": (statistics.pstdev(t_vals) if len(t_vals) > 1 else None),
            "mean_delta": (statistics.fmean(deltas) if deltas else None),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "sign_test_p_two_tailed": p,
        }
    return out


def render_md(results_per_step: dict[str, dict], axes: list[str]) -> str:
    lines: list[str] = []
    lines.append("# v3 intermediate-step PhyJudge paired-delta vs v3 baseline\n")
    for step, blob in results_per_step.items():
        lines.append(f"## step {step}  (n_paired={blob['n_paired']})")
        lines.append("")
        lines.append("| axis | baseline mean ± std | trained mean ± std | mean Δ | + | − | = | sign-test p |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for axis in axes:
            a = blob["axes"].get(axis, {})

            def _fmt(x: float | None, nd: int = 3) -> str:
                return "n/a" if x is None else f"{x:.{nd}f}"

            line = (
                f"| {axis} "
                f"| {_fmt(a.get('baseline_mean'))} ± {_fmt(a.get('baseline_std'))} "
                f"| {_fmt(a.get('trained_mean'))} ± {_fmt(a.get('trained_std'))} "
                f"| {_fmt(a.get('mean_delta'))} "
                f"| {a.get('wins', 0)} | {a.get('losses', 0)} | {a.get('ties', 0)} "
                f"| {_fmt(a.get('sign_test_p_two_tailed'), 4)} |"
            )
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--baseline-results", type=Path, required=True,
                   help="v3 full-eval results.jsonl (baseline rows are filtered).")
    p.add_argument("--run-root", type=Path, required=True,
                   help="v3_intermediate_<ts> root mirrored on the score box.")
    p.add_argument("--steps", default="50 100 150",
                   help="space-separated step ids")
    p.add_argument("--axes", default="SA PTV persistence inertia momentum")
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if not args.baseline_results.is_file():
        print(f"FATAL: baseline results not found: {args.baseline_results}", file=sys.stderr)
        return 2
    if not args.run_root.is_dir():
        print(f"FATAL: run root not found: {args.run_root}", file=sys.stderr)
        return 2

    axes = args.axes.split()
    steps = args.steps.split()

    baseline_rows = _read_jsonl(args.baseline_results)
    baseline_by_prompt = _index_by_prompt(baseline_rows, want_mode="baseline")
    if not baseline_by_prompt:
        print(
            f"FATAL: no baseline rows resolved from {args.baseline_results} "
            "(could not extract prompt_id from video paths)",
            file=sys.stderr,
        )
        return 2
    print(f"[baseline] {len(baseline_by_prompt)} prompt_ids loaded", flush=True)

    out: dict = {}
    for step in steps:
        scores_dir = args.run_root / "scores" / f"step{step}"
        results_path = _newest_results_jsonl(scores_dir)
        if results_path is None:
            print(f"[step {step}] no results.jsonl under {scores_dir}; skipping", flush=True)
            continue
        trained_rows = _read_jsonl(results_path)
        trained_by_prompt = _index_by_prompt(trained_rows, want_mode="trained")
        if not trained_by_prompt:
            print(
                f"[step {step}] no trained rows resolved from {results_path}; "
                "skipping (check video path layout)",
                flush=True,
            )
            continue
        print(
            f"[step {step}] {len(trained_by_prompt)} trained prompt_ids "
            f"from {results_path}",
            flush=True,
        )
        out[step] = compare_step(baseline_by_prompt, trained_by_prompt, axes)

    md = render_md(out, axes)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"[md] wrote {args.out_md}", flush=True)
    else:
        print(md)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[json] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
