from __future__ import annotations

from typing import Any

import torch


def dtype_name(dtype: torch.dtype | str | None) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return dtype
    text = str(dtype)
    if text.startswith("torch."):
        return text[len("torch.") :]
    return text


def tensor_dtype_name(tensor: torch.Tensor) -> str:
    return dtype_name(tensor.dtype) or "unknown"


def parse_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected dtype string, got {type(value).__name__}")
    key = value.removeprefix("torch.").lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if key in mapping:
        return mapping[key]
    if hasattr(torch, key):
        candidate = getattr(torch, key)
        if isinstance(candidate, torch.dtype):
            return candidate
    raise ValueError(f"Unsupported dtype: {value}")


def parse_w8a8_scale_dtype(value: Any) -> torch.dtype:
    """Parse a supported W8A8 weight-scale dtype."""
    dtype = parse_dtype(value)
    if dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError(
            "W8A8 scale dtype must be float32/fp32 or bfloat16/bf16"
        )
    return dtype
