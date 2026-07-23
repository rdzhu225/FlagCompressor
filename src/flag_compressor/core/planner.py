from __future__ import annotations

from flag_compressor.core.compressed_tensors import validate_fusion_closure
from flag_compressor.core.plan import ExecutionPlan, FormatSpec
from flag_compressor.core.policy import QuantizationPolicy, UnselectedWeightsPolicy
from flag_compressor.core.profile import ModelProfile, TensorInfo


def _is_scaled_fp4(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.storage_format == "fp4_e2m1_e8m0"
    )


def _is_scaled_fp8(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.storage_format == "fp8_block_e8m0"
    )


def _is_float_weight(tensor: TensorInfo) -> bool:
    return (
        tensor.role == "weight"
        and tensor.scale_name is None
        and tensor.dtype in {"bfloat16", "float16", "float32"}
        and len(tensor.shape) == 2
    )


def _input_format_for(tensor: TensorInfo) -> FormatSpec | None:
    if _is_scaled_fp4(tensor):
        return FormatSpec("fp4_e2m1_e8m0")
    if _is_scaled_fp8(tensor):
        return FormatSpec("fp8_block_e8m0", {"block_size": 128})
    if _is_float_weight(tensor):
        return FormatSpec(
            {
                "bfloat16": "bf16",
                "float16": "fp16",
                "float32": "fp32",
            }[tensor.dtype]
        )
    return None


def _bf16_output() -> FormatSpec:
    return FormatSpec("bf16")


def _plan_unselected_weight(
    plan: ExecutionPlan,
    tensor: TensorInfo,
    policy: UnselectedWeightsPolicy,
) -> None:
    """Plan one unselected weight using an inference-safe strategy.

    New strategies belong here only after their checkpoint metadata and
    runtime loader contract are implemented end to end.
    """
    if policy.strategy != "convert" or policy.format != "bf16":
        requested = (
            policy.strategy
            if policy.format is None
            else f"{policy.strategy}:{policy.format}"
        )
        raise ValueError(
            f"Unsupported unselected-weight strategy {requested!r}. "
            "Preserving a source format requires a runtime config exporter "
            "and a compatible inference kernel."
        )

    if _is_scaled_fp4(tensor):
        plan.add_action(
            tensor,
            input_format=FormatSpec("fp4_e2m1_e8m0"),
            output_format=_bf16_output(),
            rule_name="unselected_convert_bf16",
        )
    elif _is_scaled_fp8(tensor):
        plan.add_action(
            tensor,
            input_format=FormatSpec("fp8_block_e8m0", {"block_size": 128}),
            output_format=_bf16_output(),
            rule_name="unselected_convert_bf16",
        )
    else:
        plan.kept_tensors.append(tensor)
        if (
            tensor.element_size == 1
            and tensor.scale_name
            and tensor.storage_format is None
        ):
            plan.unmatched_quantized_tensors.append(tensor)


def build_convert_plan(profile: ModelProfile) -> ExecutionPlan:
    """Build a dequantization plan.

    All scaled FP8/FP4 weights are converted to BF16. Every other tensor is
    copied through unchanged.
    """
    plan = ExecutionPlan(metadata={"command": "convert", "artifact_kind": "bf16"})
    for tensor in profile.tensors.values():
        if tensor.role != "weight":
            continue
        if _is_scaled_fp4(tensor):
            plan.add_action(
                tensor,
                input_format=FormatSpec("fp4_e2m1_e8m0"),
                output_format=_bf16_output(),
            )
        elif _is_scaled_fp8(tensor):
            plan.add_action(
                tensor,
                input_format=FormatSpec("fp8_block_e8m0", {"block_size": 128}),
                output_format=_bf16_output(),
            )
        else:
            plan.kept_tensors.append(tensor)
            if tensor.element_size == 1 and tensor.scale_name:
                plan.unmatched_quantized_tensors.append(tensor)
    return plan


def build_quantize_plan(
    profile: ModelProfile,
    policy: QuantizationPolicy,
) -> ExecutionPlan:
    plan = ExecutionPlan(
        metadata={
            "command": "quantize",
            "artifact_kind": "compressed_tensors",
            "unselected_weights": {
                "strategy": policy.unselected.strategy,
                "format": policy.unselected.format,
            },
            "algorithm": {
                "name": (
                    "mse_grid_search"
                    if policy.method == "mse"
                    else policy.method
                ),
                "group_size": policy.group_size,
                "n_candidates": policy.n_candidates,
                "chunk_size": policy.chunk_size,
            },
        }
    )
    for tensor in profile.tensors.values():
        if tensor.role != "weight":
            continue

        if policy.selects(tensor):
            input_format = _input_format_for(tensor)
            if input_format is None:
                raise ValueError(
                    f"Selected tensor {tensor.name!r} cannot be quantized to INT4 "
                    f"(dtype={tensor.dtype}, storage_format={tensor.storage_format}, shape={tensor.shape})"
                )
            logical_shape = tensor.effective_logical_shape
            if len(logical_shape) != 2 or logical_shape[1] % policy.group_size:
                raise ValueError(
                    f"Selected tensor {tensor.name!r} has logical shape {logical_shape}; "
                    f"in_features must be divisible by group_size={policy.group_size}"
                )
            plan.add_action(
                tensor,
                input_format=input_format,
                output_format=FormatSpec(
                    "compressed_tensors_int4_groupwise",
                    {
                        "quantizer": policy.method,
                        "group_size": policy.group_size,
                        "n_candidates": policy.n_candidates,
                        "chunk_size": policy.chunk_size,
                    },
                ),
                rule_name="user_selected_int4",
            )
        else:
            _plan_unselected_weight(plan, tensor, policy.unselected)
    validate_fusion_closure(
        (
            tensor.name
            for tensor in profile.tensors.values()
            if tensor.role == "weight" and tensor.name.endswith(".weight")
        ),
        (
            action.tensor.name
            for action in plan.actions
            if action.output_format.name
            == "compressed_tensors_int4_groupwise"
        ),
    )
    return plan


def build_plan(profile: ModelProfile) -> ExecutionPlan:
    """Backwards-compatible alias for the BF16 conversion planner."""
    return build_convert_plan(profile)
