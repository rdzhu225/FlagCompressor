from __future__ import annotations

from quant_engine.backends.base import QuantBackend
from quant_engine.backends.cpu import CpuBackend
from quant_engine.backends.cuda import CudaBackend


def build_backend(name: str = "cpu", device: str | None = None, fallback_policy: str = "warn") -> QuantBackend:
    name = name.lower()
    if name == "cpu":
        return CpuBackend(fallback_policy)
    if name == "cuda":
        return CudaBackend(device, fallback_policy)
    raise ValueError(f"Unknown backend: {name}")
