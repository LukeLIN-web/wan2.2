"""Trainer-side runtime watchdog.

Three independent live tails — VRAM peak, DPO loss components, and the
routing counter — plus a thin ``Watchdog`` aggregator that wires them
together with one ``run_dir`` and one ``rank``. The aggregator gives the
trainer a single attach point so the integration diff stays small (see
README.md for the canonical hook into ``train_dpo_i2v.py``).

Each module writes its own JSONL under ``<run_dir>/watchdog/rank<n>/``
(line-buffered + flushed) so ``tail -f`` is a viable monitoring path
while training is in flight. End-of-run summaries are aggregated into
``<run_dir>/watchdog/rank<n>/summary.json`` and surfaced via
``Watchdog.summary()`` for inclusion in the run manifest under the
existing AC-6 manifest fields.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

from .loss_components import LossComponentLogger
from .routing_counter import LowNoiseRoutingError, RoutingCounterTail
from .vram_probe import VramProbe

__all__ = [
    "LossComponentLogger",
    "LowNoiseRoutingError",
    "RoutingCounterTail",
    "VramProbe",
    "Watchdog",
]


class Watchdog:
    """Single-attach aggregator over the three live tails.

    The trainer creates one ``Watchdog(run_dir=..., rank=rank, ...)`` per
    rank, drives ``start_step`` / ``stage`` / ``end_step`` around the
    forwards in its main loop, and calls ``log_routing`` and ``log_loss``
    at the natural recording points. ``flush_summary`` is called once at
    end of run and returns a dict suitable for splicing into the
    run-manifest JSON.

    The aggregator is rank-aware: writes go under
    ``<run_dir>/watchdog/rank<n>/{vram,loss,routing}.jsonl`` and
    ``summary.json``. Multi-rank summary aggregation is left to a
    post-run script (``humanize/dpo_v0/watchdog/aggregate.py``) so the
    online path stays cheap.
    """

    def __init__(
        self,
        run_dir: pathlib.Path,
        rank: int = 0,
        halt_on_low_noise: bool = True,
        device=None,
        enabled: bool = True,
    ) -> None:
        self.run_dir = pathlib.Path(run_dir)
        self.rank = rank
        self.enabled = enabled
        self.dir = self.run_dir / "watchdog" / f"rank{rank}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vram = VramProbe(
            out_path=self.dir / "vram.jsonl",
            device=device,
            rank=rank,
            enabled=enabled,
        )
        self.loss = LossComponentLogger(
            out_path=self.dir / "loss.jsonl",
            rank=rank,
        )
        self.routing = RoutingCounterTail(
            out_path=self.dir / "routing.jsonl",
            rank=rank,
            halt_on_low_noise=halt_on_low_noise,
        )

    # -- step lifecycle (VRAM stages) --

    def start_step(self, step: int, **meta) -> None:
        if not self.enabled:
            return
        self.vram.start_step(step, **meta)

    def stage(self, label: str) -> None:
        if not self.enabled:
            return
        self.vram.stage(label)

    def end_step(self) -> Optional[dict]:
        if not self.enabled:
            return None
        return self.vram.end_step()

    # -- per-event recording --

    def log_routing(
        self,
        step: int,
        sampled_timestep_id: int,
        raw_timestep: int,
        pair_id: str = "",
    ) -> dict:
        return self.routing.log(
            step=step,
            sampled_timestep_id=sampled_timestep_id,
            raw_timestep=raw_timestep,
            pair_id=pair_id,
        )

    def log_loss(
        self,
        step: int,
        pair_id: str,
        t_raw: int,
        loss,
        beta: float,
        components: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        return self.loss.log(
            step=step,
            pair_id=pair_id,
            t_raw=t_raw,
            loss=loss,
            beta=beta,
            components=components,
            extra=extra,
        )

    # -- summary --

    def summary(self) -> dict:
        return {
            "rank": self.rank,
            "watchdog_dir": str(self.dir),
            "vram": self.vram.summary(),
            "loss": self.loss.summary(),
            "routing": self.routing.summary(),
        }

    def flush_summary(self) -> dict:
        s = self.summary()
        (self.dir / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True), encoding="utf-8")
        return s
