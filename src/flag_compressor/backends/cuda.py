from __future__ import annotations

from flag_compressor.backends.base import QuantBackend


class CudaBackend(QuantBackend):
    def __init__(
        self,
        device: str | None = None,
        fallback_policy: str = "warn",
        op_placement: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            name="cuda",
            device_name=device or "cuda:0",
            fallback_policy=fallback_policy,
            op_placement=op_placement or {},
        )

