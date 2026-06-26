from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from quant_engine.backends.op_registry import GLOBAL_OP_REGISTRY
from quant_engine.core.report import ConversionReport, OpEvent


@dataclass
class BackendRunContext:
    report: ConversionReport | None = None


@dataclass
class QuantBackend:
    name: str
    device_name: str = "cpu"
    fallback_policy: str = "warn"
    op_placement: dict[str, str] = field(default_factory=dict)
    runtime_imports: tuple[str, ...] = ()

    def is_available(self) -> bool:
        for module_name in self.runtime_imports:
            if importlib.util.find_spec(module_name) is None:
                return False
        if self.device_name == "cpu":
            return True
        if self.device_name.startswith("cuda"):
            return torch.cuda.is_available()
        return True

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)

    def move(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device_name == "cpu":
            return tensor.cpu()
        return tensor.to(self.device)

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu()

    def synchronize(self) -> None:
        if self.device_name.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        if self.device_name.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def supports_op(self, op_name: str) -> bool:
        return GLOBAL_OP_REGISTRY.has(op_name, self.name) or GLOBAL_OP_REGISTRY.has(op_name, "torch")

    def run(
        self,
        op_name: str,
        *args: Any,
        context: BackendRunContext | None = None,
        **kwargs: Any,
    ) -> Any:
        placement = self.op_placement.get(op_name, "device")
        requested_backend = self.name if placement != "cpu" else "cpu"
        backend_available = self.is_available() if requested_backend != "cpu" else True

        impl = None
        actual_backend = requested_backend
        fallback = False
        reason = None

        if requested_backend != "cpu" and backend_available:
            impl = GLOBAL_OP_REGISTRY.resolve(op_name, self.name)
            if impl is None:
                impl = GLOBAL_OP_REGISTRY.resolve(op_name, "torch")
                actual_backend = self.name if impl is not None else self.name
        elif requested_backend != "cpu" and not backend_available:
            fallback = True
            reason = f"backend '{self.name}' is not available"

        if impl is None and requested_backend == "cpu":
            impl = GLOBAL_OP_REGISTRY.resolve(op_name, "cpu")
            actual_backend = "cpu"

        if impl is None:
            cpu_impl = GLOBAL_OP_REGISTRY.resolve(op_name, "cpu")
            if cpu_impl is None:
                raise NotImplementedError(f"No implementation registered for op '{op_name}'")
            if self.fallback_policy == "error":
                raise RuntimeError(f"Op '{op_name}' is not available on backend '{self.name}'")
            impl = cpu_impl
            actual_backend = "cpu"
            fallback = True
            reason = f"op '{op_name}' unavailable on backend '{self.name}'"

        run_args = args
        if actual_backend == "cpu":
            run_args = tuple(arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args)
        elif placement != "cpu":
            run_args = tuple(self.move(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)

        started = time.perf_counter()
        try:
            result = impl.func(*run_args, **kwargs)
        except Exception as exc:
            if actual_backend == "cpu" or self.fallback_policy == "error":
                raise
            cpu_impl = GLOBAL_OP_REGISTRY.resolve(op_name, "cpu")
            if cpu_impl is None:
                raise
            fallback = True
            reason = f"{type(exc).__name__}: {exc}"
            actual_backend = "cpu"
            run_args = tuple(arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args)
            result = cpu_impl.func(*run_args, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if context and context.report:
            context.report.record_op(
                OpEvent(
                    op_name=op_name,
                    requested_backend=requested_backend,
                    actual_backend=actual_backend,
                    fallback=fallback,
                    reason=reason,
                    elapsed_ms=elapsed_ms,
                )
            )

        return result
