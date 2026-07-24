from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from flagos_compressor.core.profile import TensorInfo


@dataclass(frozen=True)
class FormatSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorAction:
    tensor: TensorInfo
    input_format: FormatSpec
    output_format: FormatSpec
    rule_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tensor"] = asdict(self.tensor)
        return data


@dataclass
class ExecutionPlan:
    actions: list[TensorAction] = field(default_factory=list)
    kept_tensors: list[TensorInfo] = field(default_factory=list)
    unmatched_quantized_tensors: list[TensorInfo] = field(default_factory=list)
    input_format_counts: dict[str, int] = field(default_factory=dict)
    output_format_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_action(
        self,
        tensor: TensorInfo,
        *,
        input_format: FormatSpec,
        output_format: FormatSpec,
        rule_name: str | None = None,
    ) -> None:
        self.actions.append(
            TensorAction(
                tensor=tensor,
                input_format=input_format,
                output_format=output_format,
                rule_name=rule_name,
            )
        )
        self.input_format_counts[input_format.name] = (
            self.input_format_counts.get(input_format.name, 0) + 1
        )
        self.output_format_counts[output_format.name] = (
            self.output_format_counts.get(output_format.name, 0) + 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "actions": [action.to_dict() for action in self.actions],
            "kept_tensors": [asdict(t) for t in self.kept_tensors],
            "unmatched_quantized_tensors": [asdict(t) for t in self.unmatched_quantized_tensors],
            "input_format_counts": self.input_format_counts,
            "output_format_counts": self.output_format_counts,
            "metadata": self.metadata,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "action_tensors": len(self.actions),
            "kept_tensors": len(self.kept_tensors),
            "unmatched_quantized_tensors": len(self.unmatched_quantized_tensors),
            "input_formats": self.input_format_counts,
            "output_formats": self.output_format_counts,
            "input_storage_formats": dict(
                Counter(action.tensor.storage_format or "unknown" for action in self.actions)
            ),
        }
