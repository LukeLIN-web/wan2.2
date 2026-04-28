"""Per-step / per-stage CUDA VRAM peak probe.

Wraps ``torch.cuda.max_memory_allocated`` and ``torch.cuda.max_memory_reserved``
with a ``reset_peak_memory_stats`` cycle around each training step so the
peak captures only the work inside that step (not a cumulative high-water
mark from prior steps). Per-stage tags inside a step are recorded as
intermediate snapshots between forwards.

Output: one JSON line per step appended to a live JSONL file the operator
can ``tail -f`` while training. Designed to be cheap (< 100 us per call,
no graph allocation, all stats cached on the device side by the CUDA
caching allocator).

The peak number reflects the local rank only. In a multi-rank run the
aggregator opens one JSONL per rank under ``<run_dir>/watchdog/rank<n>/``.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - exercised only outside the trainer venv
    torch = None  # type: ignore[assignment]


_GIB = 1024 ** 3


class VramProbe:
    """Tracks per-step + per-stage peak VRAM in GiB.

    Usage::

        probe = VramProbe(out_path=run_dir / "watchdog" / "rank0" / "vram.jsonl")
        for step in ...:
            probe.start_step(step, pair_id=pair_id)
            # ... forward ref winner ...
            probe.stage("ref_forward_winner")
            # ... forward ref loser ...
            probe.stage("ref_forward_loser")
            # ... forward policy winner ...
            probe.stage("policy_forward_winner")
            # ... forward policy loser ...
            probe.stage("policy_forward_loser")
            # ... backward ...
            probe.stage("backward")
            probe.end_step()
    """

    def __init__(
        self,
        out_path: pathlib.Path,
        device: "Optional[torch.device]" = None,
        rank: int = 0,
        enabled: bool = True,
    ) -> None:
        self.out_path = pathlib.Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.rank = rank
        self.enabled = enabled and torch is not None and torch.cuda.is_available()
        self._step: Optional[int] = None
        self._step_t0: Optional[float] = None
        self._step_meta: dict = {}
        self._stage_log: list[dict] = []

    # -- step lifecycle --

    def start_step(self, step: int, **meta) -> None:
        if not self.enabled:
            return
        torch.cuda.reset_peak_memory_stats(self.device)
        self._step = step
        self._step_t0 = time.time()
        self._step_meta = dict(meta)
        self._stage_log = []
        # Snapshot a "step_start" baseline so the JSONL records monotonic
        # delta information even when reset_peak resets to 0.
        self.stage("step_start")

    def stage(self, label: str) -> None:
        if not self.enabled or self._step is None:
            return
        alloc_b = torch.cuda.max_memory_allocated(self.device)
        reserved_b = torch.cuda.max_memory_reserved(self.device)
        cur_alloc_b = torch.cuda.memory_allocated(self.device)
        cur_reserved_b = torch.cuda.memory_reserved(self.device)
        self._stage_log.append(
            {
                "stage": label,
                "t_offset_s": round(time.time() - (self._step_t0 or time.time()), 4),
                "peak_alloc_gib": round(alloc_b / _GIB, 4),
                "peak_reserved_gib": round(reserved_b / _GIB, 4),
                "cur_alloc_gib": round(cur_alloc_b / _GIB, 4),
                "cur_reserved_gib": round(cur_reserved_b / _GIB, 4),
            }
        )

    def end_step(self) -> dict:
        if not self.enabled or self._step is None:
            return {}
        self.stage("step_end")
        record = {
            "step": self._step,
            "rank": self.rank,
            "wall_seconds": round(time.time() - (self._step_t0 or time.time()), 4),
            "stages": list(self._stage_log),
            **self._step_meta,
        }
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
        self._step = None
        self._step_t0 = None
        self._step_meta = {}
        self._stage_log = []
        return record

    # -- summary --

    def summary(self) -> dict:
        """Read back the JSONL and surface the global peak across all steps."""
        if not self.out_path.exists():
            return {"steps": 0, "global_peak_alloc_gib": 0.0, "global_peak_reserved_gib": 0.0}
        peak_alloc = 0.0
        peak_reserved = 0.0
        steps = 0
        with self.out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                steps += 1
                for s in rec.get("stages", ()):
                    peak_alloc = max(peak_alloc, s.get("peak_alloc_gib", 0.0))
                    peak_reserved = max(peak_reserved, s.get("peak_reserved_gib", 0.0))
        return {
            "steps": steps,
            "global_peak_alloc_gib": round(peak_alloc, 4),
            "global_peak_reserved_gib": round(peak_reserved, 4),
        }
