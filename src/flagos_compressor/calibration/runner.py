"""Sequential Transformers runner for AutoGPTQ- and AutoAWQ-style calibration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import torch
from torch import nn

from flagos_compressor.calibration.mappings import (
    gptq_sequential_groups,
    infer_awq_mappings,
)
from flagos_compressor.calibration.modeling import (
    forward_layer_samples,
    move_to_device,
    sanitize_kwargs,
)
from flagos_compressor.core.policy import QuantizationPolicy
from flagos_compressor.inspect.tensor_classifier import classify_weight
from flagos_compressor.packing.autoawq import AutoAWQPacked, pack_autoawq_gemm
from flagos_compressor.packing.autogptq import AutoGPTQPacked, pack_autogptq
from flagos_compressor.quantizers.awq import (
    apply_awq_clip,
    apply_awq_scale,
    pseudo_quantize_awq,
    search_awq_clip,
    search_awq_scale,
    should_skip_awq_clip,
)
from flagos_compressor.quantizers.gptq import GPTQQuantizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeQuantizedLayer:
    method: str
    packed: AutoGPTQPacked | AutoAWQPacked


def _validate_fused_moe_selection(
    layer_name: str,
    layer: nn.Module,
    selected: set[str],
) -> None:
    """A native runtime quantizes a routed expert unit as one closed set."""
    from flagos_compressor.calibration.moe import LinearExperts2D

    for root, module in layer.named_modules():
        if not isinstance(module, LinearExperts2D):
            continue
        expert_linears = {
            f"{root}.{name}" if root else name
            for name, child in module.named_modules()
            if name and isinstance(child, nn.Linear)
        }
        chosen = expert_linears & selected
        if chosen and chosen != expert_linears:
            missing = sorted(expert_linears - chosen)
            raise ValueError(
                f"Routed experts {layer_name}.{root} are partially selected; "
                "GPTQ/AWQ native runtimes require gate/up/down for every expert. "
                f"First missing modules: {missing[:3]}"
            )


def _validate_native_shapes(
    linears: dict[str, nn.Linear],
    policy: QuantizationPolicy,
) -> None:
    group_size = int(policy.group_size or -1)
    pack_factor = 32 // policy.num_bits
    for name, linear in linears.items():
        if group_size <= 0 or linear.in_features % group_size:
            raise ValueError(
                f"{name} in_features={linear.in_features} must be divisible by "
                f"group_size={group_size}"
            )
        if linear.out_features % pack_factor:
            raise ValueError(
                f"{name} out_features={linear.out_features} must be divisible by "
                f"native pack factor {pack_factor}"
            )
        if policy.method == "gptq" and linear.in_features % pack_factor:
            raise ValueError(
                f"{name} in_features={linear.in_features} must be divisible by "
                f"native pack factor {pack_factor}"
            )


def selected_linears(
    layer_name: str,
    layer: nn.Module,
    policy: QuantizationPolicy,
) -> dict[str, nn.Linear]:
    selected: dict[str, nn.Linear] = {}
    for relative_name, module in layer.named_modules():
        if not relative_name or not isinstance(module, nn.Linear):
            continue
        full_weight_name = f"{layer_name}.{relative_name}.weight"
        _, tags = classify_weight(full_weight_name)
        if policy.selects_name(full_weight_name, tags):
            selected[relative_name] = module
    return selected


@torch.no_grad()
def _run_samples(
    layer: nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    device: torch.device,
) -> None:
    layer.to(device)
    for args, kwargs in samples:
        moved_args = move_to_device(args, device)
        moved_kwargs = move_to_device(sanitize_kwargs(layer, kwargs), device)
        layer(*moved_args, **moved_kwargs)


def _forward_kwargs_for_submodule(
    module: nn.Module,
    sample_kwargs: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    kwargs = sanitize_kwargs(module, sample_kwargs)
    return move_to_device(kwargs, device)


def _capture_awq_input(
    _module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    features: dict[str, list[torch.Tensor]],
) -> None:
    value = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(value, torch.Tensor):
        value = next(
            (item for item in kwargs.values() if isinstance(item, torch.Tensor)),
            None,
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Could not capture canonical AWQ input for {name}")
    features[name].append(value.detach().cpu())


@torch.no_grad()
def quantize_layer_gptq(
    layer_name: str,
    layer: nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    policy: QuantizationPolicy,
    *,
    device: torch.device,
) -> dict[str, NativeQuantizedLayer]:
    linears = selected_linears(layer_name, layer, policy)
    if not linears:
        return {}
    _validate_native_shapes(linears, policy)
    _validate_fused_moe_selection(layer_name, layer, set(linears))
    groups = (
        gptq_sequential_groups(list(linears))
        if policy.gptq.true_sequential
        else [list(linears)]
    )
    results: dict[str, NativeQuantizedLayer] = {}
    layer.to(device)
    for group in groups:
        quantizers = {
            name: GPTQQuantizer(
                linears[name].weight,
                bits=policy.num_bits,
                symmetric=policy.gptq.symmetric,
            )
            for name in group
        }
        handles = [
            linears[name].register_forward_pre_hook(
                lambda _module, args, q=quantizers[name]: q.add_batch(args[0])
            )
            for name in group
        ]
        try:
            _run_samples(layer, samples, device)
        finally:
            for handle in handles:
                handle.remove()
        for name in group:
            linear = linears[name]
            result = quantizers[name].quantize(
                block_size=policy.gptq.block_size,
                damp_percent=policy.gptq.damp_percent,
                group_size=int(policy.group_size or -1),
                desc_act=policy.gptq.desc_act,
                static_groups=policy.gptq.static_groups,
            )
            linear.weight.data.copy_(result.weight)
            packed = pack_autogptq(
                result.weight,
                result.scales,
                result.zeros,
                result.g_idx,
                bits=policy.num_bits,
                scale_dtype=linear.weight.dtype,
            )
            results[f"{layer_name}.{name}"] = NativeQuantizedLayer("gptq", packed)
    return results


@torch.no_grad()
def quantize_layer_awq(
    layer_name: str,
    layer: nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    policy: QuantizationPolicy,
    *,
    device: torch.device,
) -> dict[str, NativeQuantizedLayer]:
    linears = selected_linears(layer_name, layer, policy)
    if not linears:
        return {}
    _validate_native_shapes(linears, policy)
    _validate_fused_moe_selection(layer_name, layer, set(linears))
    layer.to(device)
    modules = dict(layer.named_modules())
    mappings = infer_awq_mappings(layer, set(linears))
    feature_names = set(linears)
    feature_names.update(
        mapping.input_name for mapping in mappings if mapping.input_name in modules
    )
    features: dict[str, list[torch.Tensor]] = {
        name: [] for name in sorted(feature_names)
    }
    handles = [
        modules[name].register_forward_pre_hook(
            lambda module, args, kwargs, name=name: _capture_awq_input(
                module,
                args,
                kwargs,
                name=name,
                features=features,
            ),
            with_kwargs=True,
        )
        for name in sorted(feature_names)
    ]
    try:
        _run_samples(layer, samples, device)
    finally:
        for handle in handles:
            handle.remove()
    inputs = {name: torch.cat(values, dim=0) for name, values in features.items() if values}

    for mapping in mappings:
        if mapping.input_name not in inputs:
            continue
        previous = modules[mapping.previous_name]
        search_linears = [linears[name] for name in mapping.quantized_names]
        balance = [modules[name] for name in mapping.linear_names]
        inspect_module = modules[mapping.inspect_name]
        kwargs = (
            _forward_kwargs_for_submodule(inspect_module, samples[0][1], device)
            if mapping.inspect_name.rsplit(".", 1)[-1] in {"self_attn", "attention", "attn", "linear_attn"}
            else {}
        )
        scales = search_awq_scale(
            inspect_module,
            search_linears,
            inputs[mapping.input_name],
            kwargs=kwargs,
            group_size=int(policy.group_size or 128),
            zero_point=policy.awq.zero_point,
            duo_scaling=policy.awq.duo_scaling,
            n_grid=policy.awq.n_grid,
            max_chunk_memory=policy.awq.max_chunk_memory,
        )
        apply_awq_scale(previous, balance, scales)
        for name in mapping.linear_names:
            if name in inputs:
                inputs[name].div_(scales.view(1, -1))

    if policy.awq.apply_clip:
        for name, linear in linears.items():
            if name not in inputs or should_skip_awq_clip(name):
                continue
            maximum = search_awq_clip(
                linear.weight,
                inputs[name],
                group_size=int(policy.group_size or 128),
                zero_point=policy.awq.zero_point,
                n_grid=policy.awq.n_grid,
            )
            apply_awq_clip(linear.weight, maximum)

    results: dict[str, NativeQuantizedLayer] = {}
    for name, linear in linears.items():
        quantized = pseudo_quantize_awq(
            linear.weight,
            bits=4,
            group_size=int(policy.group_size or 128),
            zero_point=policy.awq.zero_point,
        )
        if quantized.zeros is None:
            raise ValueError("Native AutoAWQ GEMM export requires zero_point=true")
        linear.weight.data.copy_(quantized.weight)
        packed = pack_autoawq_gemm(
            quantized.weight,
            quantized.scales,
            quantized.zeros,
            group_size=int(policy.group_size or 128),
            # AutoAWQ's native GEMM ABI serializes scales as FP16 even when
            # the source model itself uses BF16.
            scale_dtype=torch.float16,
        )
        results[f"{layer_name}.{name}"] = NativeQuantizedLayer("awq", packed)
    return results


@torch.no_grad()
def quantize_model_sequential(
    layers: list[tuple[str, nn.Module]],
    first_layer_samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    policy: QuantizationPolicy,
    *,
    device: torch.device,
) -> dict[str, NativeQuantizedLayer]:
    samples = first_layer_samples
    all_results: dict[str, NativeQuantizedLayer] = {}
    for layer_index, (layer_name, layer) in enumerate(layers, start=1):
        logger.info(
            "[%d/%d] %s calibration: %s",
            layer_index,
            len(layers),
            policy.method.upper(),
            layer_name,
        )
        if policy.method == "gptq":
            results = quantize_layer_gptq(
                layer_name, layer, samples, policy, device=device
            )
        elif policy.method == "awq":
            results = quantize_layer_awq(
                layer_name, layer, samples, policy, device=device
            )
        else:
            raise ValueError(f"Sequential runner does not support {policy.method}")
        all_results.update(results)
        samples = forward_layer_samples(layer, samples, device=device)
        layer.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_results


__all__ = [
    "NativeQuantizedLayer",
    "quantize_layer_awq",
    "quantize_layer_gptq",
    "quantize_model_sequential",
    "selected_linears",
]
