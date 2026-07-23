from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend


@dataclass
class ArtifactResult:
    tensors: dict[str, torch.Tensor]
    generated_tensor_names: tuple[str, ...] = ()


class WeightFormat:
    name: str

    def to_canonical(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError(f"Format {self.name!r} cannot be used as an input format")

    def from_canonical(
        self,
        tensor_name: str,
        weight: torch.Tensor,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict[str, Any],
    ) -> ArtifactResult:
        raise NotImplementedError(f"Format {self.name!r} cannot be used as an output format")


WEIGHT_FORMATS: dict[str, WeightFormat] = {}


def register_weight_format(weight_format: WeightFormat) -> WeightFormat:
    WEIGHT_FORMATS[weight_format.name] = weight_format
    return weight_format


def get_weight_format(name: str) -> WeightFormat:
    try:
        return WEIGHT_FORMATS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown weight format: {name}") from exc
