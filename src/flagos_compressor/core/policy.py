from __future__ import annotations

from dataclasses import dataclass, field
import re

from flagos_compressor.core.profile import TensorInfo


BUILTIN_SELECTIONS = {
    "moe": "moe",
    "moe.routed": "moe.routed",
    "moe.shared": "moe.shared",
    "attention": "attention",
    "mlp": "mlp",
    "linear": "linear",
}


@dataclass(frozen=True)
class CalibrationPolicy:
    """Calibration inputs shared by activation-aware quantizers."""

    data: str | tuple[str, ...] | None = None
    samples: int = 128
    sequence_length: int = 512
    seed: int = 42
    split: str = "train"
    text_column: str = "text"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.samples <= 0 or self.sequence_length <= 0:
            raise ValueError("calibration samples and sequence_length must be positive")
        if not self.split or not self.text_column:
            raise ValueError("calibration split and text_column must be non-empty")
        if isinstance(self.data, tuple) and not all(
            isinstance(item, str) and item for item in self.data
        ):
            raise ValueError("calibration data entries must be non-empty strings")


@dataclass(frozen=True)
class GPTQPolicy:
    """AutoGPTQ-compatible algorithm controls."""

    block_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = True
    static_groups: bool = False
    true_sequential: bool = True
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("gptq.block_size must be positive")
        if not 0 < self.damp_percent < 1:
            raise ValueError("gptq.damp_percent must be between 0 and 1")


@dataclass(frozen=True)
class AWQPolicy:
    """AutoAWQ-compatible W4A16 GEMM controls."""

    zero_point: bool = True
    version: str = "gemm"
    duo_scaling: bool = True
    apply_clip: bool = True
    n_grid: int = 20
    max_chunk_memory: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        normalized = self.version.lower()
        object.__setattr__(self, "version", normalized)
        if normalized != "gemm":
            raise ValueError("Only AutoAWQ GEMM checkpoint format is currently supported")
        if self.n_grid <= 0 or self.max_chunk_memory <= 0:
            raise ValueError("awq.n_grid and max_chunk_memory must be positive")


@dataclass(frozen=True)
class AutoRoundPolicy:
    """Native AutoRound controls following the reference implementation."""

    iters: int = 200
    lr: float | None = None
    minmax_lr: float | None = None
    batch_size: int = 8
    gradient_accumulate_steps: int = 1
    momentum: float = 0.0
    enable_minmax_tuning: bool = True
    enable_quantized_input: bool = True

    def __post_init__(self) -> None:
        if self.iters <= 0:
            raise ValueError("autoround.iters must be positive")
        if self.lr is not None and self.lr <= 0:
            raise ValueError("autoround.lr must be positive when specified")
        if self.minmax_lr is not None and self.minmax_lr <= 0:
            raise ValueError("autoround.minmax_lr must be positive when specified")
        if self.batch_size <= 0 or self.gradient_accumulate_steps <= 0:
            raise ValueError(
                "autoround.batch_size and gradient_accumulate_steps must be positive"
            )
        if self.momentum < 0:
            raise ValueError("autoround.momentum must be non-negative")


@dataclass(frozen=True)
class UnselectedWeightsPolicy:
    """How source-quantized weights outside the selected set are handled."""

    strategy: str = "convert"
    format: str | None = "bf16"

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("unselected.strategy must be non-empty")
        if self.format is not None and not isinstance(self.format, str):
            raise ValueError("unselected.format must be a string or null")
        if self.strategy == "convert" and not self.format:
            raise ValueError("unselected.format is required for convert strategy")
        if self.strategy == "preserve" and self.format is not None:
            raise ValueError(
                "unselected.format must be omitted for preserve strategy"
            )


@dataclass(frozen=True)
class QuantizationPolicy:
    selections: tuple[str, ...] = ()
    exclude_selections: tuple[str, ...] = ()
    include_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ()
    method: str = "mse"
    format: str | None = None
    num_bits: int = 4
    activation_num_bits: int = 16
    scale_dtype: str = "float32"
    strategy: str = "group"
    group_size: int | None = None
    n_candidates: int = 200
    chunk_size: int | None = None
    calibration: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    gptq: GPTQPolicy = field(default_factory=GPTQPolicy)
    awq: AWQPolicy = field(default_factory=AWQPolicy)
    autoround: AutoRoundPolicy = field(default_factory=AutoRoundPolicy)
    unselected: UnselectedWeightsPolicy = field(
        default_factory=UnselectedWeightsPolicy
    )

    def __post_init__(self) -> None:
        scale_dtype_aliases = {
            "fp32": "float32",
            "float32": "float32",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
        }
        if self.scale_dtype not in scale_dtype_aliases:
            raise ValueError(
                "scale_dtype must be one of: fp32, float32, bf16, bfloat16"
            )
        object.__setattr__(
            self,
            "scale_dtype",
            scale_dtype_aliases[self.scale_dtype],
        )
        unknown = sorted(
            (set(self.selections) | set(self.exclude_selections)) - set(BUILTIN_SELECTIONS)
        )
        if unknown:
            raise ValueError(f"Unknown selections: {', '.join(unknown)}")
        if self.method not in {"mse", "gptq", "awq", "autoround"}:
            raise ValueError("method must be one of: mse, gptq, awq, autoround")
        expected_format = {
            "mse": "compressed-tensors",
            "gptq": "gptq",
            "awq": "awq",
            "autoround": "gptq",
        }[self.method]
        if self.format is None:
            object.__setattr__(self, "format", expected_format)
        elif self.format != expected_format:
            raise ValueError(
                f"method={self.method!r} requires format={expected_format!r}"
            )
        if self.num_bits not in (4, 8):
            raise ValueError("num_bits must be 4 or 8")
        if self.activation_num_bits not in (8, 16):
            raise ValueError("activation_num_bits must be 8 or 16")
        if self.method in {"gptq", "awq", "autoround"} and self.activation_num_bits != 16:
            raise ValueError(f"{self.method.upper()} currently supports weight-only A16")
        if self.method == "awq" and self.num_bits != 4:
            raise ValueError("AutoAWQ GEMM currently supports only 4-bit weights")
        if self.method == "awq" and not self.awq.zero_point:
            raise ValueError("Native AutoAWQ GEMM requires awq.zero_point=true")
        if self.method == "autoround" and not self.gptq.symmetric:
            raise ValueError("Native AutoRound currently requires symmetric weights")
        if self.strategy not in {"group", "channel"}:
            raise ValueError("strategy must be 'group' or 'channel'")
        if self.method in {"gptq", "awq", "autoround"} and self.strategy != "group":
            raise ValueError(f"{self.method.upper()} requires group strategy")
        if self.activation_num_bits == 8 and (
            self.num_bits != 8 or self.strategy != "channel"
        ):
            raise ValueError(
                "W8A8 requires 8-bit weights with channel strategy"
            )
        if self.strategy == "channel" and self.num_bits != 8:
            raise ValueError("channel strategy is currently supported only for INT8")
        if self.strategy == "channel" and self.group_size is not None:
            raise ValueError("group_size must be omitted for channel strategy")
        if self.strategy == "group" and self.group_size is None:
            object.__setattr__(
                self,
                "group_size",
                (
                    128
                    if self.method in {"gptq", "awq", "autoround"}
                    else (32 if self.num_bits == 4 else 128)
                ),
            )
        if self.chunk_size is None:
            object.__setattr__(
                self,
                "chunk_size",
                4096 if self.num_bits == 4 else 1024,
            )
        assert self.chunk_size is not None
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if (
            self.num_bits == 4
            and self.group_size is not None
            and self.group_size % 2
        ):
            raise ValueError("INT4 group_size must be an even integer")
        if self.n_candidates <= 0 or self.chunk_size <= 0:
            raise ValueError("n_candidates and chunk_size must be positive")
        for pattern in (*self.include_names, *self.exclude_names):
            re.compile(pattern)

    @property
    def is_w8a8(self) -> bool:
        return self.num_bits == 8 and self.activation_num_bits == 8

    def selects(self, tensor: TensorInfo) -> bool:
        if tensor.role != "weight":
            return False
        return self.selects_name(tensor.name, tensor.tags)

    def selects_name(self, name: str, tags: tuple[str, ...]) -> bool:
        """Apply the selector contract to a live Transformers module weight."""
        tags = set(tags)
        selected = any(BUILTIN_SELECTIONS[item] in tags for item in self.selections)
        selected = selected or any(re.search(pattern, name) for pattern in self.include_names)
        if not selected:
            return False
        if any(BUILTIN_SELECTIONS[item] in tags for item in self.exclude_selections):
            return False
        return not any(re.search(pattern, name) for pattern in self.exclude_names)
