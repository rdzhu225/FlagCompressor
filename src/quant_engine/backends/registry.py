from __future__ import annotations

from quant_engine.backends.base import QuantBackend
from quant_engine.backends.cpu import CpuBackend
from quant_engine.backends.cuda import CudaBackend
from quant_engine.backends.generic_torch import GenericTorchBackend
from quant_engine.core.recipe import BackendConfig


def build_backend(config: BackendConfig) -> QuantBackend:
    name = config.name.lower()
    if name == "cpu":
        return CpuBackend(config.fallback_policy, config.op_placement)
    if name == "cuda":
        return CudaBackend(config.device, config.fallback_policy, config.op_placement)
    if name in {"ascend", "npu"}:
        return GenericTorchBackend(
            name="ascend",
            device=config.device or "npu:0",
            runtime_imports=("torch_npu",),
            fallback_policy=config.fallback_policy,
            op_placement=config.op_placement,
        )
    if name in {"cambricon", "mlu"}:
        return GenericTorchBackend(
            name="cambricon",
            device=config.device or "mlu:0",
            runtime_imports=("torch_mlu",),
            fallback_policy=config.fallback_policy,
            op_placement=config.op_placement,
        )
    if name in {"kunlun", "xpu"}:
        return GenericTorchBackend(
            name="kunlun",
            device=config.device or "xpu:0",
            fallback_policy=config.fallback_policy,
            op_placement=config.op_placement,
        )
    if name in {"musa", "mthreads"}:
        return GenericTorchBackend(
            name="musa",
            device=config.device or "musa:0",
            runtime_imports=("torch_musa",),
            fallback_policy=config.fallback_policy,
            op_placement=config.op_placement,
        )
    if name in {"dcu", "rocm"}:
        return GenericTorchBackend(
            name="dcu",
            device=config.device or "cuda:0",
            fallback_policy=config.fallback_policy,
            op_placement=config.op_placement,
        )
    raise ValueError(f"Unknown backend: {config.name}")

