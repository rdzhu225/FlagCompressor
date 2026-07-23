from __future__ import annotations

from flag_compressor.backends.base import QuantBackend
from flag_compressor.backends.cpu import CpuBackend
from flag_compressor.backends.cuda import CudaBackend


def build_backend(name: str = "cpu", device: str | None = None, fallback_policy: str = "warn") -> QuantBackend:
    name = name.lower()
    if name == "cpu":
        return CpuBackend(fallback_policy)
    if name == "cuda":
        return CudaBackend(device, fallback_policy)
    raise ValueError(f"Unknown backend: {name}")
