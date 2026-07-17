from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from quant_engine.core.plan import ExecutionPlan
from quant_engine.core.profile import ModelProfile, TensorInfo


TargetMode = Literal["bf16", "int4", "moe-int4"]

TensorSelector = Callable[[TensorInfo], bool]


MOE_EXPERT_INCLUDE_PATTERN = r".*\.experts\.\d+\.(gate|up|down|w1|w2|w3)_proj\.weight$"


def _is_scaled_fp4(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.source_format == "fp4_e2m1_e8m0"
    )


def _is_scaled_fp8(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.source_format == "fp8_block_e8m0"
    )


def _is_bf16_like(tensor: TensorInfo) -> bool:
    if tensor.role != "weight":
        return False
    if tensor.scale_name:
        return False
    return tensor.dtype in {"bfloat16", "float16", "float32"}


def _compile_patterns(patterns: Iterable[str] | None) -> list[re.Pattern[str]]:
    if not patterns:
        return []
    return [re.compile(p) for p in patterns]


def make_regex_selector(
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> TensorSelector:
    """Compile include/exclude regex lists into a tensor-name selector.

    A tensor is selected iff at least one ``include`` pattern matches and
    no ``exclude`` pattern matches. If ``include`` is empty, nothing is
    selected regardless of ``exclude``.
    """
    include_res = _compile_patterns(include)
    exclude_res = _compile_patterns(exclude)

    def selector(tensor: TensorInfo) -> bool:
        if not include_res:
            return False
        if not any(pat.search(tensor.name) for pat in include_res):
            return False
        if any(pat.search(tensor.name) for pat in exclude_res):
            return False
        return True

    return selector


def _int4_transform_for(tensor: TensorInfo) -> str | None:
    if _is_scaled_fp4(tensor):
        return "fp4_to_int4"
    if _is_scaled_fp8(tensor):
        return "fp8_to_int4"
    if _is_bf16_like(tensor):
        return "bf16_to_int4"
    return None


def _add_fp4_to_bf16(plan: ExecutionPlan, tensor: TensorInfo) -> None:
    plan.add_action(
        tensor,
        rule_name="fp4_to_bf16",
        transform="fp4_to_bf16",
        group="scaled_fp4",
    )


def _add_fp8_to_bf16(plan: ExecutionPlan, tensor: TensorInfo) -> None:
    plan.add_action(
        tensor,
        rule_name="linear_fp8_to_bf16",
        transform="fp8_to_bf16",
        group="scaled_fp8_linear",
        params={"block_size": 128},
    )


def _add_to_int4(plan: ExecutionPlan, tensor: TensorInfo, transform: str) -> None:
    plan.add_action(
        tensor,
        rule_name="weight_to_int4",
        transform=transform,
        group="user_selected_int4",
        quantizer={"name": "mse", "group_size": 32, "n_candidates": 200},
        output={"scale_suffix": ".scale"},
    )


def build_plan(
    profile: ModelProfile,
    target: TargetMode = "bf16",
    int4_selector: TensorSelector | None = None,
) -> ExecutionPlan:
    """Build a lightweight conversion plan.

    Targets:
    - bf16: All scaled FP8/FP4 weights are dequantized to BF16.
    - int4: BF16 rule applies to everything by default; tensors whose names
      match ``int4_selector`` are quantized to symmetric groupwise INT4
      instead (going through BF16 first internally, regardless of source
      format).
    - moe-int4: Backwards-compatible alias for ``int4`` with a built-in
      selector that matches MoE expert linear weights.

    ``int4_selector`` takes a :class:`TensorInfo` and returns True if the
    tensor should be quantized to INT4. It is only consulted when
    ``target != "bf16"``.
    """
    if target not in {"bf16", "int4", "moe-int4"}:
        raise ValueError(f"Unsupported target: {target}")

    effective_target: TargetMode = target
    effective_selector = int4_selector
    if target == "moe-int4":
        effective_target = "int4"
        if effective_selector is None:
            effective_selector = make_regex_selector(include=[MOE_EXPERT_INCLUDE_PATTERN])

    if effective_target == "int4" and effective_selector is None:
        effective_selector = make_regex_selector(include=[])

    plan = ExecutionPlan()
    tensors = [tensor for tensor in profile.tensors.values() if tensor.role != "scale"]
    for tensor in tensors:
        int4_transform = None
        if effective_target == "int4" and effective_selector is not None and effective_selector(tensor):
            int4_transform = _int4_transform_for(tensor)

        if int4_transform is not None:
            _add_to_int4(plan, tensor, int4_transform)
        elif _is_scaled_fp4(tensor):
            _add_fp4_to_bf16(plan, tensor)
        elif _is_scaled_fp8(tensor):
            _add_fp8_to_bf16(plan, tensor)
        else:
            plan.kept_tensors.append(tensor)
            if tensor.element_size == 1 and tensor.scale_name:
                plan.unmatched_quantized_tensors.append(tensor)
    return plan
