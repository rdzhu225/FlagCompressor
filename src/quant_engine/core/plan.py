from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from quant_engine.core.profile import TensorInfo
from quant_engine.core.recipe import RuleConfig


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

    def add_action(self, tensor: TensorInfo, rule: RuleConfig, group: str | None) -> None:
        self.actions.append(
            TensorAction(
                tensor=tensor,
                rule_name=rule.name,
                transform=rule.transform,
                params=rule.params,
                quantizer=rule.quantizer,
                output=rule.output,
                group=group,
            )
        )
        self.transform_counts[rule.transform] = self.transform_counts.get(rule.transform, 0) + 1
        if group:
            self.group_counts[group] = self.group_counts.get(group, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "kept_tensors": [asdict(t) for t in self.kept_tensors],
            "unmatched_quantized_tensors": [asdict(t) for t in self.unmatched_quantized_tensors],
            "group_counts": self.group_counts,
            "transform_counts": self.transform_counts,
        }

