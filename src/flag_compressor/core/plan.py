from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from flag_compressor.core.profile import TensorInfo


@dataclass(frozen=True)
class TensorAction:
    tensor: TensorInfo
    rule_name: str
    transform: str
    params: dict[str, Any] = field(default_factory=dict)
    quantizer: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tensor"] = asdict(self.tensor)
        return data


@dataclass
class ExecutionPlan:
    actions: list[TensorAction] = field(default_factory=list)
    kept_tensors: list[TensorInfo] = field(default_factory=list)
    unmatched_quantized_tensors: list[TensorInfo] = field(default_factory=list)
    group_counts: dict[str, int] = field(default_factory=dict)
    transform_counts: dict[str, int] = field(default_factory=dict)

    def add_action(
        self,
        tensor: TensorInfo,
        *,
        rule_name: str,
        transform: str,
        group: str | None = None,
        params: dict[str, Any] | None = None,
        quantizer: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        self.actions.append(
            TensorAction(
                tensor=tensor,
                rule_name=rule_name,
                transform=transform,
                params=params or {},
                quantizer=quantizer or {},
                output=output or {},
                group=group,
            )
        )
        self.transform_counts[transform] = self.transform_counts.get(transform, 0) + 1
        if group:
            self.group_counts[group] = self.group_counts.get(group, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "actions": [action.to_dict() for action in self.actions],
            "kept_tensors": [asdict(t) for t in self.kept_tensors],
            "unmatched_quantized_tensors": [asdict(t) for t in self.unmatched_quantized_tensors],
            "group_counts": self.group_counts,
            "transform_counts": self.transform_counts,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "action_tensors": len(self.actions),
            "kept_tensors": len(self.kept_tensors),
            "unmatched_quantized_tensors": len(self.unmatched_quantized_tensors),
            "groups": self.group_counts,
            "transforms": self.transform_counts,
            "module_kinds": dict(Counter(action.tensor.module_kind or "unknown" for action in self.actions)),
            "source_formats": dict(Counter(action.tensor.source_format or "unknown" for action in self.actions)),
        }
