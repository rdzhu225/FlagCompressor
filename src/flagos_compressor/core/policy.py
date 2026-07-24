from __future__ import annotations

from dataclasses import dataclass, field
import re

from flagos_compressor.core.profile import TensorInfo


BUILTIN_SELECTIONS = {
    "moe": "moe",
    "moe.routed": "moe.routed",
    "moe.shared": "moe.shared",
    "attention": "attention",
    "mlp": "mlp",
    "linear": "linear",
}


@dataclass(frozen=True)
class UnselectedWeightsPolicy:
    """How source-quantized weights outside the selected set are handled."""

    strategy: str = "convert"
    format: str | None = "bf16"

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("unselected.strategy must be non-empty")
        if self.format is not None and not isinstance(self.format, str):
            raise ValueError("unselected.format must be a string or null")
        if self.strategy == "convert" and not self.format:
            raise ValueError("unselected.format is required for convert strategy")
        if self.strategy == "preserve" and self.format is not None:
            raise ValueError(
                "unselected.format must be omitted for preserve strategy"
            )


@dataclass(frozen=True)
class QuantizationPolicy:
    selections: tuple[str, ...] = ()
    exclude_selections: tuple[str, ...] = ()
    include_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ()
    method: str = "mse"
    group_size: int = 32
    n_candidates: int = 200
    chunk_size: int = 4096
    unselected: UnselectedWeightsPolicy = field(
        default_factory=UnselectedWeightsPolicy
    )

    def __post_init__(self) -> None:
        unknown = sorted(
            (set(self.selections) | set(self.exclude_selections)) - set(BUILTIN_SELECTIONS)
        )
        if unknown:
            raise ValueError(f"Unknown selections: {', '.join(unknown)}")
        if self.method != "mse":
            raise ValueError("Only the MSE INT4 quantizer is currently supported")
        if self.group_size <= 0 or self.group_size % 2:
            raise ValueError("INT4 group_size must be a positive even integer")
        if self.n_candidates <= 0 or self.chunk_size <= 0:
            raise ValueError("n_candidates and chunk_size must be positive")
        for pattern in (*self.include_names, *self.exclude_names):
            re.compile(pattern)

    def selects(self, tensor: TensorInfo) -> bool:
        if tensor.role != "weight":
            return False
        tags = set(tensor.tags)
        selected = any(BUILTIN_SELECTIONS[item] in tags for item in self.selections)
        selected = selected or any(re.search(pattern, tensor.name) for pattern in self.include_names)
        if not selected:
            return False
        if any(BUILTIN_SELECTIONS[item] in tags for item in self.exclude_selections):
            return False
        return not any(re.search(pattern, tensor.name) for pattern in self.exclude_names)
