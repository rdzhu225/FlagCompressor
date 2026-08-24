"""Compatibility helpers for Intel AutoRound's public configuration API.

This module deliberately does not import :mod:`auto_round` at import time. The
native FlagOS implementation remains the default; the optional package is only
loaded when a caller explicitly asks for its public entry point.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from flagos_compressor.core.policy import QuantizationPolicy


_AUTOROUND_FIELDS = {
    "algorithm",
    "act_bits",
    "batch_size",
    "bits",
    "checkpoint_format",
    "data_type",
    "dataset",
    "enable_minmax_tuning",
    "enable_quanted_input",
    "enable_quantized_input",
    "gradient_accumulate_steps",
    "group_size",
    "iters",
    "lr",
    "minmax_lr",
    "momentum",
    "nsamples",
    "provider",
    "quant_method",
    "scheme",
    "seed",
    "seqlen",
    "sym",
}


def official_autoround_available() -> bool:
    """Return whether the optional official package can be imported."""
    try:
        return importlib.util.find_spec("auto_round") is not None
    except (ImportError, ValueError):
        return False


def official_autoround_entrypoint():
    """Load the official ``AutoRound`` class only when explicitly requested."""
    try:
        module = importlib.import_module("auto_round")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Official AutoRound is optional; install auto-round only for "
            "reference execution or parity testing"
        ) from exc
    try:
        return module.AutoRound
    except AttributeError as exc:
        raise RuntimeError("Installed auto-round does not expose AutoRound") from exc


def _read_mapping(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        path = Path(source)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid AutoRound JSON config: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("AutoRound config must be a JSON object")
    nested = raw.get("quantization_config")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("quantization_config must be a JSON object")
        raw = nested
    return raw


def load_official_autoround_config(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize an official constructor/export config into known API fields."""
    raw = _read_mapping(source)
    quant_method = str(raw.get("quant_method", "gptq")).lower()
    if quant_method not in {"gptq", "autoround", "auto-round"}:
        raise ValueError(
            "AutoRound compatibility config must use GPTQ/AutoRound packing"
        )
    normalized = {key: raw[key] for key in _AUTOROUND_FIELDS if key in raw}
    scheme = normalized.get("scheme")
    if isinstance(scheme, Mapping):
        for key in ("bits", "act_bits", "group_size", "sym", "data_type"):
            if key in scheme:
                normalized.setdefault(key, scheme[key])
    elif isinstance(scheme, str):
        compact = scheme.upper().replace("-", "")
        match = re.fullmatch(r"W(\d+)A(\d+)", compact)
        if match:
            bits, activation_bits = (int(value) for value in match.groups())
            if bits not in {4, 8} or activation_bits != 16:
                raise ValueError(
                    "Native AutoRound compatibility currently supports W4A16/W8A16"
                )
            normalized.setdefault("bits", bits)
            normalized.setdefault("act_bits", activation_bits)
    if "enable_quantized_input" in normalized:
        normalized.setdefault(
            "enable_quanted_input",
            normalized.pop("enable_quantized_input"),
        )
    defaults = {
        "bits": 4,
        "act_bits": 16,
        "group_size": 128,
        "data_type": "int",
        "iters": 200,
        "batch_size": 8,
        "gradient_accumulate_steps": 1,
        "enable_minmax_tuning": True,
        "enable_quanted_input": True,
        "nsamples": 128,
        "seqlen": 2048,
        "seed": 42,
        "quant_method": "gptq",
        "sym": True,
    }
    for key, value in defaults.items():
        normalized.setdefault(key, value)
    if normalized["act_bits"] != 16 or normalized["bits"] not in {4, 8}:
        raise ValueError(
            "Native AutoRound compatibility currently supports W4A16/W8A16"
        )
    if normalized["data_type"] is not int and str(
        normalized["data_type"]
    ).lower() != "int":
        raise ValueError("Native AutoRound compatibility currently supports INT weights")
    return normalized


def official_autoround_kwargs(policy: QuantizationPolicy) -> dict[str, Any]:
    """Translate a native policy to the official ``AutoRound(...)`` API."""
    learning_rate = policy.autoround.lr or (1.0 / policy.autoround.iters)
    kwargs: dict[str, Any] = {
        "scheme": f"W{policy.num_bits}A16",
        "dataset": policy.calibration.data,
        "iters": policy.autoround.iters,
        "seqlen": policy.calibration.sequence_length,
        "nsamples": policy.calibration.samples,
        "batch_size": policy.autoround.batch_size,
        "gradient_accumulate_steps": (
            policy.autoround.gradient_accumulate_steps
        ),
        "seed": policy.calibration.seed,
        "lr": learning_rate,
        "minmax_lr": policy.autoround.minmax_lr or learning_rate,
        "momentum": policy.autoround.momentum,
        "enable_minmax_tuning": policy.autoround.enable_minmax_tuning,
        # The misspelling is part of AutoRound's current public API.
        "enable_quanted_input": policy.autoround.enable_quantized_input,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def build_official_autoround(
    model: Any,
    tokenizer: Any,
    policy: QuantizationPolicy,
    **overrides: Any,
) -> Any:
    """Construct the optional official implementation for parity workflows."""
    kwargs = official_autoround_kwargs(policy)
    kwargs.update(overrides)
    entrypoint = official_autoround_entrypoint()
    return entrypoint(model=model, tokenizer=tokenizer, **kwargs)


def official_autoround_export_config(
    policy: QuantizationPolicy,
) -> dict[str, Any]:
    """Build AutoRound/AutoGPTQ-compatible checkpoint metadata."""
    kwargs = official_autoround_kwargs(policy)
    kwargs.pop("dataset", None)
    kwargs.pop("scheme", None)
    return {
        "bits": policy.num_bits,
        "group_size": int(policy.group_size or -1),
        "sym": True,
        "data_type": "int",
        "provider": "flagos-compressor",
        "algorithm": "autoround",
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        **kwargs,
    }


__all__ = [
    "build_official_autoround",
    "load_official_autoround_config",
    "official_autoround_available",
    "official_autoround_entrypoint",
    "official_autoround_export_config",
    "official_autoround_kwargs",
]
