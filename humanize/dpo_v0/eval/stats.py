from __future__ import annotations

import hashlib
import json
import os
import pathlib
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import binomtest as _binomtest
    _HAS_SCIPY_BINOM = True
except Exception:  # pragma: no cover
    _HAS_SCIPY_BINOM = False


AXES_DEFAULT: List[str] = ["SA", "PTV", "persistence", "inertia", "momentum"]
N_AXES_DEFAULT: int = 5
N_RESAMPLES_DEFAULT: int = 10_000
ALPHA_DEFAULT: float = 0.05
RNG_SEED_DEFAULT: int = 0

_PROMPT_CLASS_JSON = pathlib.Path(__file__).resolve().parent / "PROMPT_CLASS.json"


def _load_prompt_class_file(path: pathlib.Path = _PROMPT_CLASS_JSON) -> Dict[str, Any]:
    """Load PROMPT_CLASS.json (pid->class A-G map + class names)."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_PC = _load_prompt_class_file()
PROMPT_CLASS: Dict[str, str] = dict(_PC["prompts"])
CLASS_NAMES: Dict[str, str] = dict(_PC["classes"])


def load_scores(jsonl_dir: pathlib.Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Walk scores_perprompt/<pid>/<run_ts>/results.jsonl tree, latest run_ts wins."""
    root = pathlib.Path(jsonl_dir)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        ts_dirs = sorted(
            (d for d in pid_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
        )
        if not ts_dirs:
            continue
        chosen = ts_dirs[-1]
        results = chosen / "results.jsonl"
        if not results.exists():
            continue
        per_axis: Dict[str, Dict[str, float]] = {}
        with results.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                axis = row.get("axis")
                role = row.get("role")
                score = row.get("score")
                if axis is None or role is None or score is None:
                    continue
                per_axis.setdefault(axis, {})[role] = float(score)
        if per_axis:
            out[pid] = per_axis
    return out


def load_baseline(jsonl_path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    """Read a baseline results.jsonl, returning {pid: {axis: score}}."""
    out: Dict[str, Dict[str, float]] = {}
    with pathlib.Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row.get("prompt_id") or row.get("pair_id")
            axis = row.get("axis")
            score = row.get("score")
            if pid is None or axis is None or score is None:
                continue
            out.setdefault(pid, {})[axis] = float(score)
    return out


def baseline_sha256(jsonl_path: pathlib.Path) -> str:
    """File-level sha256 hex digest."""
    h = hashlib.sha256()
    with pathlib.Path(jsonl_path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_delta(
    trained: Dict[str, Dict[str, Any]],
    baseline: Dict[str, Dict[str, float]],
    axes: List[str] = AXES_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Per-axis delta arrays over the intersection of pids, sorted by pid."""
    pids_t = set(trained.keys())
    pids_b = set(baseline.keys())
    missing = (pids_t ^ pids_b)
    if missing:
        warnings.warn(
            f"compute_delta: {len(missing)} pid(s) skipped (not in both): "
            f"{sorted(missing)[:5]}...",
            stacklevel=2,
        )
    pids = sorted(pids_t & pids_b)
    out: Dict[str, np.ndarray] = {}
    for axis in axes:
        vals: List[float] = []
        for pid in pids:
            t = trained[pid].get(axis)
            b = baseline[pid].get(axis)
            if t is None or b is None:
                continue
            if isinstance(t, dict):
                t = t.get("trained")
                if t is None:
                    continue
            vals.append(float(t) - float(b))
        out[axis] = np.asarray(vals, dtype=float)
    return out


def bootstrap_ci(
    deltas: np.ndarray,
    n_resamples: int = N_RESAMPLES_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean; resampling is over the input rows."""
    arr = np.asarray(deltas, dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(arr.mean())
    if n < 2:
        return (point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (point, lo, hi)


def _binom_two_sided(n_pos: int, n_trials: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value for k=n_pos in n=n_trials at H0=p."""
    if n_trials == 0:
        return 1.0
    if _HAS_SCIPY_BINOM:
        return float(_binomtest(n_pos, n_trials, p).pvalue)
    from math import comb
    obs = comb(n_trials, n_pos) * (p ** n_pos) * ((1 - p) ** (n_trials - n_pos))
    total = 0.0
    for k in range(n_trials + 1):
        pk = comb(n_trials, k) * (p ** k) * ((1 - p) ** (n_trials - k))
        if pk <= obs + 1e-15:
            total += pk
    return max(0.0, min(1.0, total))


def sign_test(deltas: np.ndarray) -> Tuple[int, int, int, float]:
    """Per-element two-sided sign test (zeros excluded from the trial count)."""
    arr = np.asarray(deltas, dtype=float)
    n_pos = int(np.sum(arr > 0))
    n_neg = int(np.sum(arr < 0))
    n_tie = int(np.sum(arr == 0))
    n = n_pos + n_neg
    if n == 0:
        return (n_pos, n_neg, n_tie, 1.0)
    k = max(n_pos, n_neg)
    p = _binom_two_sided(k, n)
    return (n_pos, n_neg, n_tie, float(p))


def within_noise(mean: float, ci_lo: float, ci_hi: float) -> bool:
    """|mean| < (ci_hi - ci_lo) / 2."""
    if any(np.isnan(x) for x in (mean, ci_lo, ci_hi)):
        return False
    halfwidth = (ci_hi - ci_lo) / 2.0
    return abs(mean) < halfwidth


def paired_sign_test(
    deltas_a: Dict[str, float],
    deltas_b: Dict[str, float],
) -> Tuple[int, int, int, float]:
    """Paired sign test on (a[pid] - b[pid]) over the pid intersection."""
    pids = sorted(set(deltas_a) & set(deltas_b))
    if not pids:
        raise ValueError("no overlapping prompt_ids between runs")
    diffs = np.asarray([deltas_a[p] - deltas_b[p] for p in pids], dtype=float)
    return sign_test(diffs)


def bonferroni_alpha(
    n_axes: int,
    n_classes: int,
    alpha: float = ALPHA_DEFAULT,
) -> float:
    """Bonferroni-corrected per-test alpha = alpha / (n_axes * n_classes)."""
    if n_axes < 1 or n_classes < 1:
        raise ValueError(f"n_axes and n_classes must be >= 1 (got {n_axes}, {n_classes})")
    return alpha / float(n_axes * n_classes)


def per_class_delta(
    deltas_by_axis: Dict[str, Dict[str, float]],
    classes: Dict[str, str] = PROMPT_CLASS,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Bucket per-axis deltas by class A-G; returns {cls: {axis: array}}."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for axis, pid_to_delta in deltas_by_axis.items():
        for pid, delta in pid_to_delta.items():
            cls = classes.get(pid)
            if cls is None:
                continue
            out.setdefault(cls, {}).setdefault(axis, []).append(float(delta))
    return {
        cls: {ax: np.asarray(v, dtype=float) for ax, v in ax_map.items()}
        for cls, ax_map in out.items()
    }


def _vs_criterion(mean: float, ci_lo: float, ci_hi: float) -> str:
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "n/a"
    if within_noise(mean, ci_lo, ci_hi):
        return "within-noise"
    return "pass" if mean > 0 else "fail"


def render_headline_table(
    deltas_by_axis: Dict[str, np.ndarray],
    n: int,
    criterion_pass_fn: Optional[Callable[[float, float, float], str]] = None,
) -> str:
    """Render howtoreport.md §二 Headline table (axes-avg only)."""
    if criterion_pass_fn is None:
        criterion_pass_fn = _vs_criterion
    if not deltas_by_axis:
        return "| metric | value | 95% CI | vs criterion |\n|---|---|---|---|\n"
    arrs = list(deltas_by_axis.values())
    m = min(a.size for a in arrs)
    if m == 0:
        axes_avg_arr = np.array([], dtype=float)
    else:
        axes_avg_arr = np.mean(np.stack([a[:m] for a in arrs], axis=0), axis=0)
    mean, lo, hi = bootstrap_ci(axes_avg_arr)
    verdict = "rolling-read-only" if n < 42 else criterion_pass_fn(mean, lo, hi)
    lines = [
        "| metric | value | 95% CI | vs criterion |",
        "|---|---|---|---|",
        f"| axes-avg Δ | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |",
    ]
    return "\n".join(lines) + "\n"


def render_per_axis_table(deltas_by_axis: Dict[str, np.ndarray]) -> str:
    """Render howtoreport.md §二 Per-axis table."""
    lines = [
        "| axis | Δ | 95% CI | sign-test p (vs 0) |",
        "|---|---|---|---|",
    ]
    for axis in AXES_DEFAULT:
        arr = deltas_by_axis.get(axis)
        if arr is None or arr.size == 0:
            lines.append(f"| {axis} | n/a | n/a | n/a |")
            continue
        mean, lo, hi = bootstrap_ci(arr)
        _, _, _, p = sign_test(arr)
        lines.append(
            f"| {axis} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_per_class_table(
    per_class: Dict[str, Dict[str, np.ndarray]],
    min_n_for_ci: int = 5,
) -> str:
    """Render howtoreport.md §二 Per-class axes-avg Δ table."""
    lines = [
        "| class | n | Δ | 95% CI |",
        "|---|---|---|---|",
    ]
    for cls in ["A", "B", "C", "D", "E", "F", "G"]:
        ax_map = per_class.get(cls)
        cls_label = f"{cls} {CLASS_NAMES.get(cls, '')}".strip()
        if ax_map is None or not ax_map:
            lines.append(f"| {cls_label} | 0 | n/a | n/a |")
            continue
        arrs = list(ax_map.values())
        m = min(a.size for a in arrs)
        if m == 0:
            lines.append(f"| {cls_label} | 0 | n/a | n/a |")
            continue
        axes_avg = np.mean(np.stack([a[:m] for a in arrs], axis=0), axis=0)
        n = axes_avg.size
        if n >= min_n_for_ci:
            mean, lo, hi = bootstrap_ci(axes_avg)
            lines.append(
                f"| {cls_label} | {n} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
            )
        else:
            mean = float(axes_avg.mean())
            lines.append(f"| {cls_label} | {n} | {mean:+.3f} | n/a |")
    return "\n".join(lines) + "\n"


def render_run_identity(
    ckpt: str,
    ckpt_sha: str,
    eval_gen_out: str,
    baseline_match: bool,
    n: int,
    baseline_ref: str,
    trainer_health: str = "",
) -> str:
    """Render howtoreport.md §二 Run identity table."""
    lines = [
        "| field | value |",
        "|---|---|",
        f"| ckpt | {ckpt} |",
        f"| ckpt sha256 | {ckpt_sha} |",
        f"| trainer step health | {trainer_health} |",
        f"| eval gen out | {eval_gen_out} |",
        f"| baseline_sha256_match | {'true' if baseline_match else 'false'} |",
        f"| n | {n} |",
        f"| baseline ref | {baseline_ref} |",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="howtoreport.md §二 markdown generator"
    )
    p.add_argument("--scores-dir", required=True, type=pathlib.Path,
                   help="<gen_out>/scores_perprompt/")
    p.add_argument("--baseline", required=True, type=pathlib.Path,
                   help="baseline results.jsonl")
    p.add_argument("--output-md", required=True, type=pathlib.Path)
    p.add_argument("--ckpt", default="<unset>")
    p.add_argument("--ckpt-sha", default="<unset>")
    p.add_argument("--baseline-ref", default="<unset>")
    p.add_argument("--trainer-health", default="")
    args = p.parse_args(argv)

    trained = load_scores(args.scores_dir)
    baseline = load_baseline(args.baseline)
    deltas = compute_delta(trained, baseline)
    n = min((a.size for a in deltas.values()), default=0)
    baseline_sha = baseline_sha256(args.baseline)

    deltas_pid_by_axis: Dict[str, Dict[str, float]] = {}
    pids_sorted = sorted(set(trained) & set(baseline))
    for axis in AXES_DEFAULT:
        d: Dict[str, float] = {}
        for pid in pids_sorted:
            t = trained[pid].get(axis)
            b = baseline[pid].get(axis)
            if t is None or b is None:
                continue
            if isinstance(t, dict):
                t = t.get("trained")
                if t is None:
                    continue
            d[pid] = float(t) - float(b)
        deltas_pid_by_axis[axis] = d
    pcd = per_class_delta(deltas_pid_by_axis)

    verdict = "rolling-read-only" if n < 42 else "pass-or-fail"
    body = []
    body.append(f"# Round-? eval (n={n})\n")
    body.append(f"**Verdict**: {verdict}\n")
    body.append("**Criterion**: <fill from plan>\n")
    body.append("**Next action**: <fill>\n\n")
    body.append("## Run identity\n\n")
    body.append(render_run_identity(
        ckpt=args.ckpt, ckpt_sha=args.ckpt_sha,
        eval_gen_out=str(args.scores_dir),
        baseline_match=True, n=n, baseline_ref=args.baseline_ref,
        trainer_health=args.trainer_health,
    ))
    body.append(f"\nbaseline_sha256: {baseline_sha}\n\n")
    body.append(f"## Headline (n={n})\n\n")
    body.append(render_headline_table(deltas, n))
    body.append(f"\n## Per-axis Δ (n={n})\n\n")
    body.append(render_per_axis_table(deltas))
    body.append("\n## Per-class Δ axes-avg\n\n")
    body.append(render_per_class_table(pcd))

    args.output_md.write_text("".join(body), encoding="utf-8")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
