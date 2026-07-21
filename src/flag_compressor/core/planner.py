from __future__ import annotations

from flag_compressor.core.plan import ExecutionPlan
from flag_compressor.core.profile import ModelProfile, TensorInfo


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


def build_plan(profile: ModelProfile) -> ExecutionPlan:
    """Build a dequantization plan.

    All scaled FP8/FP4 weights are converted to BF16. Every other tensor is
    copied through unchanged.
    """
    plan = ExecutionPlan()
    for tensor in profile.tensors.values():
        if tensor.role == "scale":
            continue
        if _is_scaled_fp4(tensor):
            plan.add_action(tensor, transform="fp4_to_bf16")
        elif _is_scaled_fp8(tensor):
            plan.add_action(tensor, transform="fp8_to_bf16", params={"block_size": 128})
        else:
            plan.kept_tensors.append(tensor)
            if tensor.element_size == 1 and tensor.scale_name:
                plan.unmatched_quantized_tensors.append(tensor)
    return plan
