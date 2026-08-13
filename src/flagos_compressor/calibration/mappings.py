"""Lightweight structure mappings; model forward remains owned by Transformers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class AWQMapping:
    previous_name: str
    quantized_names: tuple[str, ...]
    linear_names: tuple[str, ...]
    inspect_name: str
    input_name: str


def gptq_sequential_groups(names: list[str]) -> list[list[str]]:
    """Llama-like ordering with a deterministic catch-all for new architectures."""
    stages = (
        ("q_proj", "k_proj", "v_proj", "qkv_proj", "query_key_value", "in_proj_"),
        ("o_proj", "out_proj"),
        ("gate_proj", "up_proj", "gate_up_proj", "fc1", "w1", "w3"),
        ("down_proj", "fc2", "w2"),
    )
    remaining = list(names)
    groups: list[list[str]] = []
    for suffixes in stages:
        group = [
            name
            for name in remaining
            if any(name.rsplit(".", 1)[-1].startswith(suffix) for suffix in suffixes)
        ]
        if group:
            groups.append(group)
            remaining = [name for name in remaining if name not in group]
    # Unknown projections are still quantized, but each receives a fresh forward
    # so true-sequential error propagation remains conservative.
    groups.extend([[name] for name in remaining])
    return groups


def _first_existing(modules: dict[str, nn.Module], names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in modules), None)


def _is_weight_consumer(module: nn.Module | None) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, nn.Parameter) and weight.dim() == 2


def infer_awq_mappings(layer: nn.Module, selected: set[str]) -> list[AWQMapping]:
    """Infer AutoAWQ scale relations from common Transformers module structure."""
    modules = dict(layer.named_modules())
    mappings: list[AWQMapping] = []

    attention_roots = [
        name
        for name in modules
        if name and name.rsplit(".", 1)[-1] in {"self_attn", "attention", "attn", "linear_attn"}
    ]
    for root in attention_roots:
        qkv = tuple(
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1]
            in {"q_proj", "k_proj", "v_proj", "qkv_proj", "query_key_value"}
        )
        qkv += tuple(
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1].startswith("in_proj_")
        )
        previous = _first_existing(
            modules,
            (
                "input_layernorm",
                "attention_norm",
                "attn_norm",
                "pre_attention_layernorm",
            ),
        )
        if previous and qkv:
            mappings.append(AWQMapping(previous, qkv, qkv, root, root))

        value = next(
            (
                name
                for name in sorted(selected)
                if name.endswith(".v_proj") and name.startswith(root)
            ),
            None,
        )
        if value is None:
            value = next(
                (
                    name
                    for name in sorted(selected)
                    if name.startswith(root)
                    and name.rsplit(".", 1)[-1]
                    in {"qkv_proj", "query_key_value"}
                ),
                None,
            )
        output = next(
            (
                name
                for name in sorted(selected)
                if name.startswith(root)
                and name.rsplit(".", 1)[-1] in {"o_proj", "out_proj", "dense"}
            ),
            None,
        )
        if value and output:
            mappings.append(AWQMapping(value, (output,), (output,), output, output))

    mlp_roots = [
        name
        for name in modules
        if name and name.rsplit(".", 1)[-1] in {"mlp", "feed_forward", "ffn"}
    ]
    for root in mlp_roots:
        inputs = tuple(
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1]
            in {"gate_proj", "up_proj", "gate_up_proj", "fc1", "w1", "w3"}
        )
        previous = _first_existing(
            modules,
            (
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "ffn_norm",
                "pre_mlp_layernorm",
            ),
        )
        if previous and inputs:
            # MoE routers consume the same normalized hidden states as their
            # experts. They normally stay in floating point, but must receive
            # the inverse equalization transform or routing would change.
            router_names = tuple(
                name
                for name in (
                    f"{root}.gate",
                    f"{root}.router",
                    f"{root}.shared_expert_gate",
                )
                if _is_weight_consumer(modules.get(name)) and name not in inputs
            )
            mappings.append(
                AWQMapping(
                    previous,
                    inputs,
                    inputs + router_names,
                    root,
                    root,
                )
            )

        for up_name in [
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1] in {"up_proj", "gate_up_proj", "fc1", "w3"}
        ]:
            parent = up_name.rsplit(".", 1)[0]
            down_name = next(
                (
                    f"{parent}.{leaf}"
                    for leaf in ("down_proj", "fc2", "w2")
                    if f"{parent}.{leaf}" in selected
                ),
                None,
            )
            if down_name:
                mappings.append(
                    AWQMapping(
                        up_name,
                        (down_name,),
                        (down_name,),
                        down_name,
                        down_name,
                    )
                )
    return mappings


__all__ = ["AWQMapping", "gptq_sequential_groups", "infer_awq_mappings"]
