from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from quant_engine.backends.base import BackendRunContext, QuantBackend
from quant_engine.core.plan import TensorAction


@dataclass
class TransformResult:
    tensors: dict[str, torch.Tensor]
    generated_scale_names: list[str]


class Transform:
    name: str

    def apply(
        self,
        action: TensorAction,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
    ) -> TransformResult:
        raise NotImplementedError


TRANSFORMS: dict[str, Transform] = {}


def register_transform(transform: Transform) -> Transform:
    TRANSFORMS[transform.name] = transform
    return transform


def get_transform(name: str) -> Transform:
    try:
        return TRANSFORMS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown transform: {name}") from exc

