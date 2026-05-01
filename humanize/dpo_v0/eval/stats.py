"""Canonical statistics helpers for round-N PhyJudge eval reports.

Single source of truth for the numbers that go into
``docs/experiment-results/round*_v3_*.md`` — md authors must call these
functions instead of hand-computing aggregates. See
``docs/experiment-results/howtoreport.md`` for the reporting rules these
helpers exist to support.

Conventions
-----------
- A "score record" is a dict with at least::

      {"pair_id": str, "axis": str, "score": int}  # score in {1, 2, 3, 4}

  with optional ``role`` / ``prompt_id`` keys. Aggregations are keyed by
  ``prompt_id`` (sha256(prompt)[:12]) when present, otherwise ``pair_id``.

- A "delta record" is a dict::

      {"prompt_id": str, "axis": str, "delta": float}

  i.e. trained.score - baseline.score for the same (prompt_id, axis).

- ``axes_avg`` per prompt = mean over the 5 axes
  ``["SA", "PTV", "persistence", "inertia", "momentum"]``.

- All bootstrap CIs are **prompt-level resamples** (not per-axis or
  per-pair-id resamples) — howtoreport.md §四 fixes this protocol.

The five canonical axes (and their fixed order):
"""

from __future__ import annotations

import json
import math
import pathlib
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

AXES: tuple[str, ...] = ("SA", "PTV", "persistence", "inertia", "momentum")
"""The five PhyJudge axes in canonical order. Reports must list axes in
this order and never invent additional ones."""

CLASS_A_G_PROMPT_IDS: dict[str, list[str]] = {
    # Source of truth: docs/eval/evalprompt.md §按物理现象的细分
    "A": [  # 多体碰撞 / 反弹 (n=12)
        "2455740c4d45", "24d86e4e0339", "e0dae745a2a3", "e90b3f54bffb",
        "2559ab47b909", "1a0d4f1d8b1a", "3719b41ec796", "70d3b1b89e19",
        "5f68f5951b6b", "cb4c5cd47231", "ad664fa349ef", "31cd7275ca92",
    ],
    "B": [  # 破坏 / 形变 (n=9)
        "1b1c06c5ff1c", "eef5be6cabd2", "d858e0d67470", "242e01f46c08",
        "61345a00dfb5", "8f8b14d04c41", "8d44a2958eb4", "f6cad0ea56a8",
        "58db668bc142",
    ],
    "C": [  # 流体 / 液体动力学 (n=6)
        "8b8d6d0a9919", "2be476eeac0d", "e38a4396df92", "fa37196314bf",
        "252b84def499", "1a44aba35343",
    ],
    "D": [  # 阴影 / 反射 / 光学 (n=5)
        "75f6acbf5ba7", "261fccfc811f", "7977e8df650c", "488e8d91cff5",
        "6b48a3f28874",
    ],
    "E": [  # 链式 / 多级触发 (n=3)
        "e7815fab19d6", "36e42af19937", "9d500eec2188",
    ],
    "F": [  # 滚动 / 滑动 / 持续动量 (n=4)
        "5fdbe9f87762", "48255a441729", "80cc85fa7fa7", "31ea17615154",
    ],
    "G": [  # 抛掷 / 弹道 (n=3)
        "2db7ce10fffb", "daed47f0fab3",
        # NOTE: third pid for G is missing in evalprompt.md as committed
        # (heading says n=3 but only 2 entries listed). When the third
        # is added upstream, append here. Until then class G effectively
        # n=2 — flag in caveats per howtoreport.md §一 #4 (raw Δ only,
        # no CI for n<5 anyway).
    ],
}
"""Static prompt_id → class A-G map. ``n=42`` total; class G has the
known evalprompt.md gap (n=2 in source, 3 promised in heading).

For ``n<5`` classes (E/F/G) only raw Δ is meaningful; CI is suppressed
by ``per_class_axes_avg_with_ci``.
"""


def _build_prompt_to_class() -> dict[str, str]:
    out: dict[str, str] = {}
    for cls, pids in CLASS_A_G_PROMPT_IDS.items():
        for pid in pids:
            if pid in out:
                raise RuntimeError(
                    f"duplicate prompt_id {pid} in classes {out[pid]} and {cls}"
                )
            out[pid] = cls
    return out


PROMPT_TO_CLASS: dict[str, str] = _build_prompt_to_class()


# ---------- Aggregation primitives ----------


def per_prompt_axes_avg(
    delta_records: Iterable[Mapping[str, Any]],
    axes: Sequence[str] = AXES,
) -> dict[str, float]:
    """Reduce a stream of delta records to ``{prompt_id: axes_avg}``.

    Asserts each prompt has exactly ``len(axes)`` axis entries (5 by
    default). Missing/extra axis records raise ``ValueError`` so silent
    averaging-over-incomplete-records cannot poison downstream stats.
    """
    by_prompt: dict[str, dict[str, float]] = {}
    for r in delta_records:
        pid = r["prompt_id"]
        ax = r["axis"]
        if ax not in axes:
            continue
        by_prompt.setdefault(pid, {})[ax] = float(r["delta"])

    out: dict[str, float] = {}
    n_axes = len(axes)
    for pid, ax_map in by_prompt.items():
        if len(ax_map) != n_axes:
            missing = sorted(set(axes) - set(ax_map))
            extra = sorted(set(ax_map) - set(axes))
            raise ValueError(
                f"prompt_id {pid}: expected {n_axes} axes, got "
                f"{len(ax_map)} (missing={missing}, extra={extra})"
            )
        out[pid] = sum(ax_map[a] for a in axes) / n_axes
    return out


def per_axis_deltas(
    delta_records: Iterable[Mapping[str, Any]],
    axis: str,
) -> dict[str, float]:
    """Project delta records onto a single ``axis`` → ``{prompt_id: delta}``."""
    out: dict[str, float] = {}
    for r in delta_records:
        if r["axis"] != axis:
            continue
        pid = r["prompt_id"]
        if pid in out:
            raise ValueError(
                f"duplicate ({pid}, {axis}) in delta_records (existing="
                f"{out[pid]}, new={r['delta']})"
            )
        out[pid] = float(r["delta"])
    return out


# ---------- Bootstrap CI ----------


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the **mean** of ``values``.

    Per howtoreport.md §四:
    - n_resamples = 10000 by default
    - resample is over the input rows (treat as prompt-level when caller
      passes per-prompt aggregates)
    - returns ``(point, lo, hi)`` where ``point = mean(values)`` and
      ``[lo, hi]`` is the (1-alpha) percentile interval.

    For ``len(values) < 2`` the CI half-width is undefined; we return
    ``(point, nan, nan)`` so the caller can downgrade to "raw Δ only".
    """
    n = len(values)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    point = sum(values) / n
    if n < 2:
        return (point, math.nan, math.nan)

    rng = random.Random(rng_seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample_sum = 0.0
        for _i in range(n):
            sample_sum += values[rng.randrange(n)]
        means.append(sample_sum / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1
    lo_idx = max(0, min(n_resamples - 1, lo_idx))
    hi_idx = max(0, min(n_resamples - 1, hi_idx))
    return (point, means[lo_idx], means[hi_idx])


# ---------- Sign-tests ----------


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for k successes in n trials."""
    if n == 0:
        return 1.0

    def _pmf(i: int) -> float:
        # math.comb is 3.8+; videodpoWan runs on 3.11.
        return math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))

    obs = _pmf(k)
    total = 0.0
    for i in range(n + 1):
        if _pmf(i) <= obs + 1e-15:  # tolerance for fp ties
            total += _pmf(i)
    # Clamp to [0, 1] for numerical safety.
    return max(0.0, min(1.0, total))


def sign_test(
    deltas: Sequence[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Per-element two-sided sign-test of ``delta != 0``.

    Zeros are excluded (consistent with the standard sign-test
    convention; PhyJudge integer scale produces exact zeros at non-trivial
    rate so this matters). Reports ``n_pos``, ``n_neg``, ``n_zero``,
    ``p_two_sided``, and ``reject = p < alpha``.
    """
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)
    n = n_pos + n_neg
    if n == 0:
        return {
            "n_pos": 0, "n_neg": 0, "n_zero": n_zero,
            "p_two_sided": 1.0, "reject": False, "alpha": alpha,
        }
    k = max(n_pos, n_neg)
    p = _binom_two_sided_p(k, n)
    return {
        "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
        "p_two_sided": p, "reject": p < alpha, "alpha": alpha,
    }


def paired_sign_test(
    deltas_run_a: Mapping[str, float],
    deltas_run_b: Mapping[str, float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired sign-test on prompt-id-aligned Δ values.

    For each prompt that appears in **both** mappings, computes
    ``run_a[pid] - run_b[pid]`` and runs ``sign_test`` on the differences.
    Returns the standard sign-test dict augmented with ``n_aligned`` and
    ``aligned_pids`` for audit trail.

    This is the canonical winner-comparison per howtoreport.md §四:
    "比较 round 间 winner: 按相同 prompt id 集合配对 sign-test, 不允许
    直接比 axes-avg 点估计".
    """
    pids = sorted(set(deltas_run_a) & set(deltas_run_b))
    if not pids:
        raise ValueError("no overlapping prompt_ids between runs")
    diffs = [deltas_run_a[p] - deltas_run_b[p] for p in pids]
    out = sign_test(diffs, alpha=alpha)
    out["n_aligned"] = len(pids)
    out["aligned_pids"] = pids
    return out


# ---------- Multi-comparison (Bonferroni) ----------


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Bonferroni-corrected per-test alpha. ``n_tests = 5 * N_class`` in
    the standard "5 axes × N classes" report layout per howtoreport.md
    §四."""
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1 (got {n_tests})")
    return alpha / float(n_tests)


# ---------- Per-class breakdown ----------


def per_class_axes_avg_with_ci(
    per_prompt_axes_avg_map: Mapping[str, float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    min_n_for_ci: int = 5,
    rng_seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Class-A-G axes-avg roll-up with bootstrap CI.

    For each class with ``n >= min_n_for_ci`` prompts present in the input
    map, returns ``{cls: {"n": int, "delta": float, "ci_lo": float,
    "ci_hi": float}}``. Below threshold, ``ci_lo`` / ``ci_hi`` are
    ``None`` (per howtoreport.md §二 "n<5 的 class 只显示 raw Δ").

    Classes with zero overlap are omitted entirely.
    """
    out: dict[str, dict[str, Any]] = {}
    for cls, pids in CLASS_A_G_PROMPT_IDS.items():
        present = [per_prompt_axes_avg_map[p] for p in pids
                   if p in per_prompt_axes_avg_map]
        if not present:
            continue
        n = len(present)
        if n >= min_n_for_ci:
            point, lo, hi = bootstrap_ci(
                present, n_resamples=n_resamples,
                alpha=alpha, rng_seed=rng_seed,
            )
            out[cls] = {"n": n, "delta": point,
                        "ci_lo": lo, "ci_hi": hi}
        else:
            out[cls] = {"n": n,
                        "delta": sum(present) / n,
                        "ci_lo": None, "ci_hi": None}
    return out


# ---------- Loader for results.jsonl ----------


def load_delta_records_from_results(
    trained_results_path: pathlib.Path | str,
    baseline_results_path: pathlib.Path | str,
    score_field: str = "score",
) -> list[dict[str, Any]]:
    """Read trained + baseline results.jsonl files and emit aligned
    delta records.

    Each input file is JSONL with rows of shape
    ``{"prompt_id": ..., "pair_id": ..., "axis": ..., "score": int, ...}``.
    The merge key is ``(prompt_id, axis)``. Pairs in trained but not in
    baseline (and vice versa) are dropped silently — caller can verify
    cardinality before calling.
    """
    def _load(p: pathlib.Path | str) -> dict[tuple[str, str], dict[str, Any]]:
        path = pathlib.Path(p)
        out: dict[tuple[str, str], dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = row.get("prompt_id") or row.get("pair_id")
                ax = row["axis"]
                key = (pid, ax)
                if key in out:
                    raise ValueError(
                        f"duplicate ({pid}, {ax}) in {path}"
                    )
                out[key] = row
        return out

    trained = _load(trained_results_path)
    baseline = _load(baseline_results_path)
    records: list[dict[str, Any]] = []
    for key in sorted(trained.keys() & baseline.keys()):
        pid, ax = key
        records.append({
            "prompt_id": pid,
            "axis": ax,
            "delta": float(trained[key][score_field])
                     - float(baseline[key][score_field]),
            "trained_score": float(trained[key][score_field]),
            "baseline_score": float(baseline[key][score_field]),
        })
    return records


# ---------- One-shot summary for a report ----------


def summarize_run(
    delta_records: Sequence[Mapping[str, Any]],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Compute the full set of numbers that the howtoreport.md template
    asks for.

    Returns a dict with:
    - ``axes_avg``: ``{"delta": ..., "ci_lo": ..., "ci_hi": ..., "n": ...}``
      — point + CI on the per-prompt axes-avg.
    - ``per_axis``: ``{axis: {"delta", "ci_lo", "ci_hi", "sign_test_p"}}``.
    - ``per_class``: ``{cls: {"n", "delta", "ci_lo", "ci_hi"}}`` — A-G.
    - ``n_prompts``: number of prompts.
    - ``axes_avg_sign_test``: sign-test on per-prompt Δ-axes-avg vs 0.

    The caller is responsible for then comparing against a pre-registered
    criterion (``paired_sign_test`` against another run's same dict).
    """
    pp = per_prompt_axes_avg(delta_records)
    n_prompts = len(pp)
    axes_avg_values = list(pp.values())
    axes_avg_point, axes_avg_lo, axes_avg_hi = bootstrap_ci(
        axes_avg_values, n_resamples=n_resamples,
        alpha=alpha, rng_seed=rng_seed,
    )
    axes_avg_signtest = sign_test(axes_avg_values, alpha=alpha)

    per_axis: dict[str, dict[str, Any]] = {}
    for ax in AXES:
        ax_deltas_map = per_axis_deltas(delta_records, ax)
        ax_values = list(ax_deltas_map.values())
        point, lo, hi = bootstrap_ci(
            ax_values, n_resamples=n_resamples,
            alpha=alpha, rng_seed=rng_seed,
        )
        st = sign_test(ax_values, alpha=alpha)
        per_axis[ax] = {
            "delta": point, "ci_lo": lo, "ci_hi": hi,
            "sign_test_p": st["p_two_sided"], "n": len(ax_values),
        }

    per_class = per_class_axes_avg_with_ci(
        pp, n_resamples=n_resamples, alpha=alpha, rng_seed=rng_seed,
    )

    return {
        "n_prompts": n_prompts,
        "axes_avg": {
            "delta": axes_avg_point,
            "ci_lo": axes_avg_lo, "ci_hi": axes_avg_hi,
            "sign_test_p": axes_avg_signtest["p_two_sided"],
            "n_pos": axes_avg_signtest["n_pos"],
            "n_neg": axes_avg_signtest["n_neg"],
            "n_zero": axes_avg_signtest["n_zero"],
        },
        "per_axis": per_axis,
        "per_class": per_class,
    }


# ---------- Tiny CLI for ad-hoc use ----------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Compute round-N PhyJudge eval stats per howtoreport.md."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize",
                           help="trained vs baseline summary")
    p_sum.add_argument("--trained-results", required=True,
                       type=pathlib.Path)
    p_sum.add_argument("--baseline-results", required=True,
                       type=pathlib.Path)
    p_sum.add_argument("--n-resamples", type=int, default=10_000)
    p_sum.add_argument("--alpha", type=float, default=0.05)
    p_sum.add_argument("--rng-seed", type=int, default=0)
    p_sum.add_argument("--out-json", type=pathlib.Path, default=None)

    p_paired = sub.add_parser("paired-sign-test",
                              help="aligned-pid sign-test of run-a vs run-b")
    p_paired.add_argument("--run-a-trained", required=True,
                          type=pathlib.Path,
                          help="trained results.jsonl for run A")
    p_paired.add_argument("--run-a-baseline", required=True,
                          type=pathlib.Path)
    p_paired.add_argument("--run-b-trained", required=True,
                          type=pathlib.Path)
    p_paired.add_argument("--run-b-baseline", required=True,
                          type=pathlib.Path)
    p_paired.add_argument("--alpha", type=float, default=0.05)
    p_paired.add_argument("--out-json", type=pathlib.Path, default=None)

    args = p.parse_args(argv)

    if args.cmd == "summarize":
        records = load_delta_records_from_results(
            args.trained_results, args.baseline_results,
        )
        out = summarize_run(
            records, n_resamples=args.n_resamples,
            alpha=args.alpha, rng_seed=args.rng_seed,
        )
        text = json.dumps(out, indent=2, ensure_ascii=False)
        if args.out_json:
            args.out_json.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    if args.cmd == "paired-sign-test":
        a_records = load_delta_records_from_results(
            args.run_a_trained, args.run_a_baseline,
        )
        b_records = load_delta_records_from_results(
            args.run_b_trained, args.run_b_baseline,
        )
        a_pp = per_prompt_axes_avg(a_records)
        b_pp = per_prompt_axes_avg(b_records)
        out = paired_sign_test(a_pp, b_pp, alpha=args.alpha)
        # aligned_pids may be long; keep it but write to file separately
        # if the caller used --out-json.
        text = json.dumps(out, indent=2, ensure_ascii=False)
        if args.out_json:
            args.out_json.write_text(text + "\n", encoding="utf-8")
        # When dumping to stdout, suppress the long pid list for skim.
        compact = {k: v for k, v in out.items() if k != "aligned_pids"}
        compact["n_aligned"] = out["n_aligned"]
        print(json.dumps(compact, indent=2, ensure_ascii=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
