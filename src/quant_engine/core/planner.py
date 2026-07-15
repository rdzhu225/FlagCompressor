from __future__ import annotations

from typing import Literal

from quant_engine.core.plan import ExecutionPlan
from quant_engine.core.profile import ModelProfile, TensorInfo


TargetMode = Literal["bf16", "moe-int4"]


def _is_scaled_fp4(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.source_format == "fp4_e2m1_e8m0"
    )


def _is_moe_fp4(tensor: TensorInfo) -> bool:
    return _is_scaled_fp4(tensor) and tensor.module_kind == "moe_mlp_linear"


def _is_scaled_fp8(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.source_format == "fp8_block_e8m0"
    )


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


def _add_moe_fp4_to_int4(plan: ExecutionPlan, tensor: TensorInfo) -> None:
    plan.add_action(
        tensor,
        rule_name="moe_fp4_to_int4",
        transform="fp4_to_int4",
        group="moe_mlp_linear",
        quantizer={"name": "mse", "group_size": 32, "n_candidates": 200},
        output={"scale_suffix": ".scale"},
    )


def build_plan(profile: ModelProfile, target: TargetMode = "bf16") -> ExecutionPlan:
    """Build the fixed lightweight conversion plan.

    Targets:
    - bf16: FP8 and FP4 scaled weights are dequantized to BF16.
    - moe-int4: FP8 is dequantized to BF16, MoE FP4 is quantized to INT4,
      and any non-MoE FP4 tensors are dequantized to BF16.
    """
    if target not in {"bf16", "moe-int4"}:
        raise ValueError(f"Unsupported target: {target}")

    plan = ExecutionPlan()
    tensors = [tensor for tensor in profile.tensors.values() if tensor.role != "scale"]
    for tensor in tensors:
        if target == "moe-int4" and _is_moe_fp4(tensor):
            _add_moe_fp4_to_int4(plan, tensor)
        elif _is_scaled_fp4(tensor):
            _add_fp4_to_bf16(plan, tensor)
        elif _is_scaled_fp8(tensor):
            _add_fp8_to_bf16(plan, tensor)
        else:
            plan.kept_tensors.append(tensor)
            if tensor.element_size == 1 and tensor.scale_name:
                plan.unmatched_quantized_tensors.append(tensor)
    return plan
