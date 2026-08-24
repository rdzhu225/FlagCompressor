from __future__ import annotations

from flagos_compressor.backends.base import QuantBackend
from flagos_compressor.backends.cpu import CpuBackend
from flagos_compressor.backends.cuda import CudaBackend


BUILTIN_DEVICE_BACKENDS = ("cpu", "cuda", "npu", "mlu", "musa")

_RUNTIME_IMPORTS = {
    "npu": ("torch_npu",),
    "mlu": ("torch_mlu",),
    "musa": ("torch_musa",),
}


def build_backend(
    name: str = "cpu",
    device: str | None = None,
    fallback_policy: str = "warn",
) -> QuantBackend:
    name = name.lower()
    if name == "cpu":
        return CpuBackend(fallback_policy)
    if name == "cuda":
        return CudaBackend(device, fallback_policy)
    return QuantBackend(
        name=name,
        device_name=device or f"{name}:0",
        fallback_policy=fallback_policy,
        runtime_imports=_RUNTIME_IMPORTS.get(name, ()),
    )


__all__ = ["BUILTIN_DEVICE_BACKENDS", "build_backend"]
