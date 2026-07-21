from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from flag_compressor.core.profile import TensorInfo


@dataclass(frozen=True)
class TensorAction:
    tensor: TensorInfo
    transform: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tensor"] = asdict(self.tensor)
        return data


@dataclass
class ExecutionPlan:
    actions: list[TensorAction] = field(default_factory=list)
    kept_tensors: list[TensorInfo] = field(default_factory=list)
    unmatched_quantized_tensors: list[TensorInfo] = field(default_factory=list)
    transform_counts: dict[str, int] = field(default_factory=dict)

    def add_action(
        self,
        tensor: TensorInfo,
        *,
        transform: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.actions.append(
            TensorAction(
                tensor=tensor,
                transform=transform,
                params=params or {},
            )
        )
        self.transform_counts[transform] = self.transform_counts.get(transform, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "actions": [action.to_dict() for action in self.actions],
            "kept_tensors": [asdict(t) for t in self.kept_tensors],
            "unmatched_quantized_tensors": [asdict(t) for t in self.unmatched_quantized_tensors],
            "transform_counts": self.transform_counts,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "action_tensors": len(self.actions),
            "kept_tensors": len(self.kept_tensors),
            "unmatched_quantized_tensors": len(self.unmatched_quantized_tensors),
            "transforms": self.transform_counts,
            "source_formats": dict(Counter(action.tensor.source_format or "unknown" for action in self.actions)),
        }
