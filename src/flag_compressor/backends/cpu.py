from __future__ import annotations

from flag_compressor.backends.base import QuantBackend


class CpuBackend(QuantBackend):
    def __init__(self, fallback_policy: str = "warn", op_placement: dict[str, str] | None = None) -> None:
        super().__init__(
            name="cpu",
            device_name="cpu",
            fallback_policy=fallback_policy,
            op_placement=op_placement or {},
        )

