from __future__ import annotations

from quant_engine.backends.base import QuantBackend


class GenericTorchBackend(QuantBackend):
    def __init__(
        self,
        name: str,
        device: str,
        runtime_imports: tuple[str, ...] = (),
        fallback_policy: str = "warn",
        op_placement: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            device_name=device,
            fallback_policy=fallback_policy,
            op_placement=op_placement or {},
            runtime_imports=runtime_imports,
        )

