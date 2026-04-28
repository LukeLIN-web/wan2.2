"""Routing-counter live tail with hard-fail on low-noise hit.

Mirrors the in-trainer ``RoutingCounter`` (train_dpo_i2v.py) but writes a
live JSONL per call so an operator can ``tail -f`` the routing log while
training is in flight. Halts on any low-noise increment per AC-5.U3
(100% high-noise contract on tier_a / tier_b).

This module is additive: the in-trainer counter is the source of truth
for the run manifest summary; this module exists so monitoring can see
each routing event the moment it happens, without waiting for the run
manifest to be written at the end.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass


# AC-5.U2 raw boundary mirror: switch_DiT_boundary * 1000 = 900.
# By Wan2.2's MoE convention, the high-noise expert is active when
# raw_timestep > 900, low-noise otherwise.
SWITCH_DIT_BOUNDARY_RAW = 900


def detect_expert(raw_timestep: int) -> str:
    return "high_noise" if int(raw_timestep) > SWITCH_DIT_BOUNDARY_RAW else "low_noise"


class LowNoiseRoutingError(RuntimeError):
    """Raised when a low-noise routing event is observed under the 100%-high-noise contract."""


@dataclass
class _Entry:
    step: int
    sampled_timestep_id: int
    raw_timestep: int
    detected_expert: str
    pair_id: str


class RoutingCounterTail:
    """JSONL-tailed routing counter.

    Usage::

        tail = RoutingCounterTail(out_path=run_dir / "watchdog" / "rank0" / "routing.jsonl",
                                  halt_on_low_noise=True)
        tail.log(step=step, sampled_timestep_id=step, raw_timestep=t_raw, pair_id=pair_id)

    The ``halt_on_low_noise`` flag mirrors the in-trainer assertion. Set
    it to False only on the eval-harness smoke (M5) where low-noise is
    the *expected* expert for the frozen low-noise sampling tail; for
    M3 / M4 training it must remain True.
    """

    def __init__(
        self,
        out_path: pathlib.Path,
        rank: int = 0,
        halt_on_low_noise: bool = True,
    ) -> None:
        self.out_path = pathlib.Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.halt_on_low_noise = halt_on_low_noise
        self.high_count = 0
        self.low_count = 0
        self._entries: list[_Entry] = []

    @property
    def total(self) -> int:
        return self.high_count + self.low_count

    def log(
        self,
        step: int,
        sampled_timestep_id: int,
        raw_timestep: int,
        pair_id: str = "",
    ) -> dict:
        expert = detect_expert(raw_timestep)
        if expert == "high_noise":
            self.high_count += 1
        else:
            self.low_count += 1
        entry = _Entry(
            step=int(step),
            sampled_timestep_id=int(sampled_timestep_id),
            raw_timestep=int(raw_timestep),
            detected_expert=expert,
            pair_id=pair_id,
        )
        self._entries.append(entry)
        record = {
            "step": entry.step,
            "rank": self.rank,
            "sampled_timestep_id": entry.sampled_timestep_id,
            "raw_timestep": entry.raw_timestep,
            "detected_expert": entry.detected_expert,
            "pair_id": entry.pair_id,
            "cum_high": self.high_count,
            "cum_low": self.low_count,
            "cum_fraction_high": round(self.high_count / max(1, self.total), 6),
            "wall_unix_s": round(time.time(), 3),
        }
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
        if expert == "low_noise" and self.halt_on_low_noise:
            raise LowNoiseRoutingError(
                f"low-noise routing detected at step={step} sampled_timestep_id={sampled_timestep_id} "
                f"raw_timestep={raw_timestep}; tier_a/tier_b contract requires 100% high-noise."
            )
        return record

    def summary(self) -> dict:
        return {
            "total_forwards": self.total,
            "high_count": self.high_count,
            "low_count": self.low_count,
            "fraction_high_noise": round(self.high_count / max(1, self.total), 6),
            "halt_on_low_noise": self.halt_on_low_noise,
        }
