from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OpEvent:
    op_name: str
    requested_backend: str
    actual_backend: str
    fallback: bool = False
    reason: str | None = None
    elapsed_ms: float | None = None


@dataclass
class ConversionReport:
    backend: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    converted: int = 0
    kept: int = 0
    skipped_scales: int = 0
    output_tensors: int = 0
    op_events: list[OpEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def record_op(self, event: OpEvent) -> None:
        self.op_events.append(event)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def finish(self) -> None:
        self.finished_at = time.time()

    def op_summary(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "fallback": 0, "backends": defaultdict(int), "elapsed_ms": 0.0}
        )
        for event in self.op_events:
            item = summary[event.op_name]
            item["count"] += 1
            item["backends"][event.actual_backend] += 1
            if event.fallback:
                item["fallback"] += 1
            if event.elapsed_ms is not None:
                item["elapsed_ms"] += event.elapsed_ms
        return {
            op: {
                **values,
                "backends": dict(values["backends"]),
                "elapsed_ms": round(values["elapsed_ms"], 3),
            }
            for op, values in summary.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": None if self.finished_at is None else self.finished_at - self.started_at,
            "converted": self.converted,
            "kept": self.kept,
            "skipped_scales": self.skipped_scales,
            "output_tensors": self.output_tensors,
            "ops": self.op_summary(),
            "op_events": [asdict(event) for event in self.op_events],
            "warnings": self.warnings,
            "extras": self.extras,
        }

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

