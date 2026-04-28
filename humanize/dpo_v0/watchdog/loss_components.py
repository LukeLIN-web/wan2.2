"""Per-step loss + DPO components live tail.

Records the four MSE components, policy / reference advantage, raw DPO
logit, and implied policy-over-reference winner probability returned by
``flow_matching_dpo_loss(..., return_components=True)``.

Output: one JSON line per step appended to a live JSONL file. The file
is operator-tailable (line-buffered + flushed). All component tensors
are reduced to a Python float here — caller passes either pre-reduced
floats or 0-D tensors.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Mapping, Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# Field set covers the components emitted by dpo_loss.flow_matching_dpo_loss
# (see humanize/dpo_v0/dpo_loss.py). Caller may pass a subset; missing
# fields are written as null.
KNOWN_COMPONENT_FIELDS = (
    "mse_policy_winner",
    "mse_policy_loser",
    "mse_reference_winner",
    "mse_reference_loser",
    "policy_advantage",
    "reference_advantage",
    "logit",
    "implied_winner_prob",
)


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if torch is not None and isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class LossComponentLogger:
    """Per-step JSONL writer for DPO loss components.

    Usage::

        logger = LossComponentLogger(out_path=run_dir / "watchdog" / "rank0" / "loss.jsonl")
        loss, components = flow_matching_dpo_loss(..., return_components=True)
        logger.log(
            step=step,
            pair_id=pair_id,
            t_raw=t_raw,
            loss=loss,
            beta=beta,
            components=components,
        )
    """

    def __init__(
        self,
        out_path: pathlib.Path,
        rank: int = 0,
        warn_on_nan: bool = True,
    ) -> None:
        self.out_path = pathlib.Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.warn_on_nan = warn_on_nan
        self.nan_count = 0
        self.steps_logged = 0

    def log(
        self,
        step: int,
        pair_id: str,
        t_raw: int,
        loss,
        beta: float,
        components: Optional[Mapping[str, object]] = None,
        extra: Optional[Mapping[str, object]] = None,
    ) -> dict:
        loss_val = _to_float(loss)
        record = {
            "step": int(step),
            "rank": self.rank,
            "pair_id": pair_id,
            "t_raw": int(t_raw),
            "beta": float(beta),
            "loss": loss_val,
            "wall_unix_s": round(time.time(), 3),
        }
        if components:
            for k in KNOWN_COMPONENT_FIELDS:
                record[k] = _to_float(components.get(k)) if k in components else None
            # Anything else the caller surfaced (free-form extras live alongside).
            for k, v in components.items():
                if k in KNOWN_COMPONENT_FIELDS:
                    continue
                record[f"extra__{k}"] = _to_float(v) if not isinstance(v, str) else v
        else:
            for k in KNOWN_COMPONENT_FIELDS:
                record[k] = None
        if extra:
            for k, v in extra.items():
                record[f"meta__{k}"] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

        if self.warn_on_nan and loss_val is not None and math.isnan(loss_val):
            self.nan_count += 1
            record["nan_loss"] = True

        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
        self.steps_logged += 1
        return record

    def summary(self) -> dict:
        if not self.out_path.exists():
            return {"steps_logged": 0, "nan_count": 0}
        loss_values: list[float] = []
        nan_count = 0
        with self.out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                lv = rec.get("loss")
                if lv is None:
                    continue
                if math.isnan(lv):
                    nan_count += 1
                    continue
                loss_values.append(lv)
        if not loss_values:
            return {"steps_logged": 0, "nan_count": nan_count}
        return {
            "steps_logged": len(loss_values),
            "nan_count": nan_count,
            "loss_min": round(min(loss_values), 6),
            "loss_max": round(max(loss_values), 6),
            "loss_mean": round(sum(loss_values) / len(loss_values), 6),
            "loss_first": round(loss_values[0], 6),
            "loss_last": round(loss_values[-1], 6),
        }
