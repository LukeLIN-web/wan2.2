#!/usr/bin/env python3
"""Summarize DPO training metrics from a W&B run.

This is intentionally focused on the metrics emitted by
``train/train_dpo_i2v.py``:

* ``accuracy`` is the per-step sign of the DPO margin.
* ``acc_win50`` is the trainer's rolling 50-step mean of that sign.
* ``loss`` and ``margin``/``logit`` show whether the run is fitting the
  pair-preference objective, not whether the generated videos improve.

Example:

    python humanize/dpo_v0/eval/analyze_wandb_dpo_run.py \
      --run-path lukelin/wanrl/9if26kr9 \
      --token-path /shared/user60/worldmodel/rlvideo/videodpoWan/wandbtoken
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import wandb


DEFAULT_RUN_PATH = "lukelin/wanrl/9if26kr9"
DEFAULT_TOKEN_PATH = Path("/shared/user60/worldmodel/rlvideo/videodpoWan/wandbtoken")
DEFAULT_QUARTERS = "0:50,50:100,100:150,150:200"
DEFAULT_HISTORY_MODE = "quick"
DEFAULT_SAMPLES = 2000
DEFAULT_TAIL_STEPS = 200
DEFAULT_WINDOW_SCAN_KEYS = (
    "loss",
    "accuracy",
    "acc_win50",
    "margin",
    "logit",
    "grad_norm",
    "grad_finite",
)
DEFAULT_KEYS = (
    "loss",
    "accuracy",
    "acc_win50",
    "margin",
    "logit",
    "delta",
    "grad_norm",
    "grad_finite",
    "chosen_reward",
    "rejected_reward",
    "chosen_logp",
    "rejected_logp",
    "mse_pi_w",
    "mse_pi_l",
    "mse_ref_w",
    "mse_ref_l",
    "c_w",
    "t_raw",
    "elapsed_s",
)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def numeric_stats(values: Iterable[Any]) -> dict[str, float | int] | None:
    vals = [v for v in (finite_float(value) for value in values) if v is not None]
    if not vals:
        return None
    out: dict[str, float | int] = {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }
    if len(vals) > 1:
        out["std"] = statistics.pstdev(vals)
    return out


def round_floats(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_floats(val, ndigits) for key, val in value.items()}
    if isinstance(value, list):
        return [round_floats(val, ndigits) for val in value]
    return value


def parse_ranges(spec: str) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"bad range {part!r}; expected START:END"
            )
        start_s, end_s = part.split(":", 1)
        try:
            start = int(start_s)
            end = int(end_s)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"bad range {part!r}; START and END must be ints"
            ) from exc
        if end <= start:
            raise argparse.ArgumentTypeError(
                f"bad range {part!r}; END must be greater than START"
            )
        ranges.append((f"{start}_{end - 1}", start, end))
    if not ranges:
        raise argparse.ArgumentTypeError("at least one range is required")
    return ranges


def compact_step(row: dict[str, Any]) -> dict[str, float | int | None]:
    margin = finite_float(row.get("margin", row.get("logit")))
    return {
        "_step": int(row["_step"]),
        "loss": finite_float(row.get("loss")),
        "accuracy": finite_float(row.get("accuracy")),
        "acc_win50": finite_float(row.get("acc_win50")),
        "margin": margin,
        "chosen_reward": finite_float(row.get("chosen_reward")),
        "rejected_reward": finite_float(row.get("rejected_reward")),
    }


def summarize_range(
    rows: list[dict[str, Any]],
    label: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    selected = [row for row in rows if start <= row["_step"] < end]
    losses = [row.get("loss") for row in selected]
    loss_count = sum(1 for value in losses if finite_float(value) is not None)
    loss_lt_0p1 = sum(
        1
        for value in losses
        if finite_float(value) is not None and finite_float(value) < 0.1
    )
    acc_stats = numeric_stats(row.get("accuracy") for row in selected)
    acc_win50_stats = numeric_stats(row.get("acc_win50") for row in selected)
    return {
        "label": label,
        "n": len(selected),
        "steps": [selected[0]["_step"], selected[-1]["_step"]] if selected else None,
        "accuracy_mean": acc_stats["mean"] if acc_stats else None,
        "acc_win50_mean": acc_win50_stats["mean"] if acc_win50_stats else None,
        "acc_win50_last": (
            finite_float(selected[-1].get("acc_win50")) if selected else None
        ),
        "margin": numeric_stats(
            row.get("margin", row.get("logit")) for row in selected
        ),
        "loss": numeric_stats(losses),
        "loss_lt_0p1_frac": loss_lt_0p1 / loss_count if loss_count else None,
        "wrong_steps": [
            row["_step"] for row in selected if finite_float(row.get("accuracy")) == 0.0
        ],
        "negative_margin_steps": [
            row["_step"]
            for row in selected
            if (
                finite_float(row.get("margin", row.get("logit"))) is not None
                and finite_float(row.get("margin", row.get("logit"))) < 0
            )
        ],
    }


def load_wandb_rows(
    run_path: str,
    token_path: Path | None,
    keys: tuple[str, ...],
    timeout: int,
    page_size: int,
    history_mode: str = DEFAULT_HISTORY_MODE,
    samples: int = DEFAULT_SAMPLES,
    tail_steps: int = DEFAULT_TAIL_STEPS,
    ranges: list[tuple[str, int, int]] | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    if token_path is not None:
        token = token_path.read_text().strip()
        if not token:
            raise SystemExit(f"empty W&B token file: {token_path}")
        os.environ["WANDB_API_KEY"] = token
    elif not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("provide --token-path or set WANDB_API_KEY")

    os.environ.setdefault("WANDB_SILENT", "true")
    api = wandb.Api(timeout=timeout)
    run = api.run(run_path)

    if history_mode == "quick":
        sampled_rows = _sample_slim_history(run=run, keys=keys, samples=samples)
        last_step = _last_history_step(run)
        tail_range = _tail_scan_range(last_step=last_step, tail_steps=tail_steps)
        tail_rows = (
            _scan_slim_history(
                run=run,
                keys=keys,
                page_size=page_size,
                scan_keys=keys,
                min_step=tail_range[0],
                max_step=tail_range[1],
            )
            if tail_range is not None
            else []
        )
        if tail_range is not None and not tail_rows:
            tail_rows = _scan_slim_history(
                run=run,
                keys=keys,
                page_size=page_size,
                scan_keys=DEFAULT_WINDOW_SCAN_KEYS,
                min_step=tail_range[0],
                max_step=tail_range[1],
            )
        rows = _merge_slim_rows(sampled_rows, tail_rows)
        history_info = {
            "mode": history_mode,
            "complete": False,
            "samples_requested": samples,
            "sampled_rows": len(sampled_rows),
            "tail_steps": tail_steps,
            "tail_rows": len(tail_rows),
            "tail_range": (
                {"start": tail_range[0], "end_exclusive": tail_range[1]}
                if tail_range is not None
                else None
            ),
            "last_history_step": last_step,
            "note": (
                "Stats are based on sampled history plus an exact tail window; "
                "use --history-mode full for complete scan_history stats."
            ),
        }
    elif history_mode == "range-scan":
        scan_ranges = _requested_scan_ranges(
            run=run,
            ranges=ranges or [],
            tail_steps=tail_steps,
        )
        rows = _scan_slim_ranges(
            run=run,
            keys=keys,
            page_size=page_size,
            scan_ranges=scan_ranges,
        )
        history_info = {
            "mode": history_mode,
            "complete": False,
            "scan_ranges": [
                {"start": start, "end_exclusive": end}
                for start, end in scan_ranges
            ],
            "tail_steps": tail_steps,
            "note": (
                "Stats are exact only within requested scan ranges; "
                "use --history-mode full for complete scan_history stats."
            ),
        }
    elif history_mode == "full":
        rows = _scan_slim_history(
            run=run,
            keys=keys,
            page_size=page_size,
            scan_keys=keys,
        )
        history_info = {
            "mode": history_mode,
            "complete": True,
            "scan": "scan_history(keys=...)",
        }
        if not rows:
            print(
                (
                    "[wandb] scan_history(keys=...) returned 0 rows; "
                    "falling back to full history scan and local key filtering."
                ),
                file=sys.stderr,
            )
            rows = _scan_slim_history(
                run=run,
                keys=keys,
                page_size=page_size,
                scan_keys=None,
            )
            history_info["scan"] = "scan_history()"
    else:
        raise SystemExit(
            "bad --history-mode; expected one of: quick, range-scan, full"
        )

    if not rows:
        raise SystemExit(f"no W&B history rows found for {run_path}")
    return run, rows, history_info


def _sample_slim_history(
    run: Any,
    keys: tuple[str, ...],
    samples: int,
) -> list[dict[str, Any]]:
    if samples <= 0:
        return []
    rows = run.history(samples=samples, keys=list(keys), pandas=False)
    return _merge_slim_rows(_slim_rows(rows, keys))


def _scan_slim_history(
    run: Any,
    keys: tuple[str, ...],
    page_size: int,
    scan_keys: tuple[str, ...] | None,
    min_step: int | None = None,
    max_step: int | None = None,
) -> list[dict[str, Any]]:
    scan_kwargs: dict[str, Any] = {"page_size": page_size}
    if scan_keys is not None:
        scan_kwargs["keys"] = _scan_keys_with_step(scan_keys)
    if min_step is not None:
        scan_kwargs["min_step"] = min_step
    if max_step is not None:
        scan_kwargs["max_step"] = max_step
    rows = _slim_rows(run.scan_history(**scan_kwargs), keys)
    if min_step is not None or max_step is not None:
        rows = [
            row
            for row in rows
            if (
                (min_step is None or row["_step"] >= min_step)
                and (max_step is None or row["_step"] < max_step)
            )
        ]
    return _merge_slim_rows(rows)


def _scan_slim_ranges(
    run: Any,
    keys: tuple[str, ...],
    page_size: int,
    scan_ranges: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start, end in scan_ranges:
        range_rows = _scan_slim_history(
            run=run,
            keys=keys,
            page_size=page_size,
            scan_keys=keys,
            min_step=start,
            max_step=end,
        )
        if not range_rows:
            range_rows = _scan_slim_history(
                run=run,
                keys=keys,
                page_size=page_size,
                scan_keys=DEFAULT_WINDOW_SCAN_KEYS,
                min_step=start,
                max_step=end,
            )
        rows.extend(range_rows)
    return _merge_slim_rows(rows)


def _scan_keys_with_step(keys: tuple[str, ...]) -> list[str]:
    scan_keys = ["_step"]
    for key in keys:
        if key != "_step" and key not in scan_keys:
            scan_keys.append(key)
    return scan_keys


def _slim_rows(
    rows: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    slim_rows: list[dict[str, Any]] = []
    for row in rows:
        slim = _slim_row(row, keys)
        if slim is not None:
            slim_rows.append(slim)
    return slim_rows


def _slim_row(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any] | None:
    step = row.get("_step")
    if step is None:
        return None
    slim: dict[str, Any] = {"_step": int(step)}
    for key in keys:
        if key in row:
            slim[key] = row[key]
    return slim


def _merge_slim_rows(*row_groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_step: dict[int, dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            step = int(row["_step"])
            merged = rows_by_step.setdefault(step, {"_step": step})
            for key, value in row.items():
                if key == "_step":
                    continue
                merged[key] = value
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def _last_history_step(run: Any) -> int | None:
    value = finite_float(getattr(run, "lastHistoryStep", None))
    if value is None:
        summary = getattr(run, "summary", {}) or {}
        value = finite_float(summary.get("_step"))
    return int(value) if value is not None else None


def _tail_scan_range(
    last_step: int | None,
    tail_steps: int,
) -> tuple[int, int] | None:
    if last_step is None or tail_steps <= 0:
        return None
    end = last_step + 1
    start = max(0, end - tail_steps)
    return start, end


def _requested_scan_ranges(
    run: Any,
    ranges: list[tuple[str, int, int]],
    tail_steps: int,
) -> list[tuple[int, int]]:
    last_step = _last_history_step(run)
    upper = last_step + 1 if last_step is not None else None
    scan_ranges: list[tuple[int, int]] = []
    for _, start, end in ranges:
        start = max(0, start)
        if upper is not None:
            end = min(end, upper)
        if end > start:
            scan_ranges.append((start, end))
    tail_range = _tail_scan_range(last_step=last_step, tail_steps=tail_steps)
    if tail_range is not None:
        scan_ranges.append(tail_range)
    if not scan_ranges and upper is not None:
        scan_ranges.append((max(0, upper - max(tail_steps, 1)), upper))
    return _merge_ranges(scan_ranges)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def build_summary(
    run: Any,
    rows: list[dict[str, Any]],
    ranges: list[tuple[str, int, int]],
    top_k: int,
    history_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    margin_rows = [
        row for row in rows if finite_float(row.get("margin", row.get("logit"))) is not None
    ]
    loss_rows = [row for row in rows if finite_float(row.get("loss")) is not None]

    top_loss = sorted(
        loss_rows,
        key=lambda row: finite_float(row.get("loss")) or float("-inf"),
        reverse=True,
    )[:top_k]
    most_negative_margin = sorted(
        margin_rows,
        key=lambda row: finite_float(row.get("margin", row.get("logit")))
        or float("inf"),
    )[:top_k]
    most_positive_margin = sorted(
        margin_rows,
        key=lambda row: finite_float(row.get("margin", row.get("logit")))
        or float("-inf"),
        reverse=True,
    )[:top_k]

    summary_keys = (
        "loss",
        "accuracy",
        "acc_win50",
        "margin",
        "grad_norm",
        "_step",
        "_runtime",
    )
    return {
        "run": {
            "path": run.path,
            "name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
        },
        "history": history_info or {"mode": "unknown", "complete": None},
        "row_count": len(rows),
        "step_range": [rows[0]["_step"], rows[-1]["_step"]],
        "summary_subset": {
            key: run.summary.get(key) for key in summary_keys if key in run.summary
        },
        "final_row": rows[-1],
        "ranges": {
            label: summarize_range(rows, label, start, end)
            for label, start, end in ranges
        },
        "overall": {
            "loss": numeric_stats(row.get("loss") for row in rows),
            "accuracy": numeric_stats(row.get("accuracy") for row in rows),
            "acc_win50": numeric_stats(row.get("acc_win50") for row in rows),
            "margin": numeric_stats(row.get("margin", row.get("logit")) for row in rows),
            "chosen_reward": numeric_stats(row.get("chosen_reward") for row in rows),
            "rejected_reward": numeric_stats(row.get("rejected_reward") for row in rows),
            "mse_pi_w": numeric_stats(row.get("mse_pi_w") for row in rows),
            "mse_pi_l": numeric_stats(row.get("mse_pi_l") for row in rows),
            "mse_ref_w": numeric_stats(row.get("mse_ref_w") for row in rows),
            "mse_ref_l": numeric_stats(row.get("mse_ref_l") for row in rows),
            "grad_norm": numeric_stats(row.get("grad_norm") for row in rows),
        },
        "top_loss_steps": [compact_step(row) for row in top_loss],
        "most_negative_margin_steps": [
            compact_step(row) for row in most_negative_margin
        ],
        "most_positive_margin_steps": [
            compact_step(row) for row in most_positive_margin
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize train_dpo_i2v.py metrics from a W&B run.",
    )
    parser.add_argument("--run-path", default=DEFAULT_RUN_PATH)
    parser.add_argument(
        "--token-path",
        default=str(DEFAULT_TOKEN_PATH),
        help=(
            "W&B API token file. Use --token-path env to rely on WANDB_API_KEY."
        ),
    )
    parser.add_argument(
        "--quarters",
        type=parse_ranges,
        default=parse_ranges(DEFAULT_QUARTERS),
        help="Comma-separated step ranges START:END, END exclusive.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument(
        "--history-mode",
        choices=("quick", "range-scan", "full"),
        default=DEFAULT_HISTORY_MODE,
        help=(
            "quick uses sampled history plus an exact tail scan; range-scan "
            "scans only --quarters and tail; full scans the entire run."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Sample count for --history-mode quick.",
    )
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=DEFAULT_TAIL_STEPS,
        help="Exact trailing step window to scan for quick/range-scan modes.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_path = None
    if args.token_path not in {"", "env", "none", "-"}:
        token_path = Path(args.token_path)
    run, rows, history_info = load_wandb_rows(
        run_path=args.run_path,
        token_path=token_path,
        keys=DEFAULT_KEYS,
        timeout=args.timeout,
        page_size=args.page_size,
        history_mode=args.history_mode,
        samples=args.samples,
        tail_steps=args.tail_steps,
        ranges=args.quarters,
    )
    summary = build_summary(
        run=run,
        rows=rows,
        ranges=args.quarters,
        top_k=args.top_k,
        history_info=history_info,
    )
    print(json.dumps(round_floats(summary), indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()
