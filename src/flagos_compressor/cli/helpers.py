from __future__ import annotations

import json
from pathlib import Path

from flagos_compressor.core.plan import ExecutionPlan
from flagos_compressor.core.policy import QuantizationPolicy, UnselectedWeightsPolicy


def print_plan(plan: ExecutionPlan) -> None:
    print("Execution plan")
    algorithm = plan.metadata.get("algorithm") or {}
    if algorithm:
        detail = (
            f"W{algorithm.get('num_bits')}A"
            f"{algorithm.get('activation_num_bits', 16)} "
            f"{algorithm.get('strategy')} "
            f"{algorithm.get('name')}"
        )
        if algorithm.get("group_size") is not None:
            detail += f" group_size={algorithm['group_size']}"
        print(f"  quantization: {detail}")
    print("  input formats")
    for input_format, count in sorted(plan.input_format_counts.items()):
        print(f"    {input_format}: {count}")
    print("  output formats")
    for output_format, count in sorted(plan.output_format_counts.items()):
        print(f"    {output_format}: {count}")
    print(f"  keep: {len(plan.kept_tensors)}")
    print(f"  unmatched quantized: {len(plan.unmatched_quantized_tensors)}")


def ensure_no_unmatched(plan: ExecutionPlan) -> None:
    if not plan.unmatched_quantized_tensors:
        return
    examples = ", ".join(t.name for t in plan.unmatched_quantized_tensors[:3])
    raise RuntimeError(
        f"Found {len(plan.unmatched_quantized_tensors)} scaled byte tensors with unsupported "
        f"layouts; refusing to guess their formats. Examples: {examples}"
    )


def load_quantize_recipe(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Recipe support requires PyYAML; install flagos-compressor dependencies"
        ) from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Quantize recipe must be a YAML mapping")
    allowed = {
        "version",
        "bits",
        "activation_bits",
        "strategy",
        "method",
        "group_size",
        "n_candidates",
        "chunk_size",
        "select",
        "exclude",
        "unselected",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown recipe keys: {', '.join(unknown)}")
    if data.get("version", 1) != 1:
        raise ValueError(f"Unsupported recipe version: {data.get('version')}")
    return data


def _parse_selector_items(items) -> tuple[list[str], list[str]]:
    groups: list[str] = []
    names: list[str] = []
    for item in items or []:
        if isinstance(item, str):
            groups.append(item)
        elif isinstance(item, dict) and set(item) == {"name"} and isinstance(item["name"], str):
            names.append(item["name"])
        else:
            raise ValueError("Selectors must be built-in names or mappings like {name: 'REGEX'}")
    return groups, names


def _parse_unselected_policy(value) -> UnselectedWeightsPolicy:
    if value is None:
        return UnselectedWeightsPolicy()
    if not isinstance(value, dict):
        raise ValueError("unselected must be a mapping")
    unknown = sorted(set(value) - {"strategy", "format"})
    if unknown:
        raise ValueError(f"Unknown unselected keys: {', '.join(unknown)}")
    strategy = value.get("strategy", "convert")
    target_format = value.get(
        "format",
        None if strategy == "preserve" else "bf16",
    )
    return UnselectedWeightsPolicy(
        strategy=strategy,
        format=target_format,
    )


def build_quantization_policy(args) -> QuantizationPolicy:
    recipe = load_quantize_recipe(args.recipe) if args.recipe else {}

    recipe_groups, recipe_names = _parse_selector_items(recipe.get("select"))
    exclude_groups, recipe_excludes = _parse_selector_items(recipe.get("exclude"))
    selections = tuple(recipe_groups + list(args.select or ()))
    exclude_selections = tuple(exclude_groups + list(args.exclude or ()))
    include_names = tuple(recipe_names + list(args.select_name or ()))
    exclude_names = tuple(recipe_excludes + list(args.exclude_name or ()))
    if not selections and not include_names:
        raise ValueError(
            "No weights selected; use --select/--select-name or a recipe"
        )

    def value(name: str, default):
        cli_value = getattr(args, name, None)
        return cli_value if cli_value is not None else recipe.get(name, default)

    num_bits = int(value("bits", 4))
    activation_num_bits = int(value("activation_bits", 16))
    strategy = value("strategy", "group")
    requested_group_size = value("group_size", None)
    if strategy == "group" and requested_group_size is None:
        requested_group_size = 32 if num_bits == 4 else 128
    default_chunk_size = 4096 if num_bits == 4 else 1024
    return QuantizationPolicy(
        selections=selections,
        exclude_selections=exclude_selections,
        include_names=include_names,
        exclude_names=exclude_names,
        method=value("method", "mse"),
        num_bits=num_bits,
        activation_num_bits=activation_num_bits,
        strategy=strategy,
        group_size=(
            int(requested_group_size)
            if requested_group_size is not None
            else None
        ),
        n_candidates=int(value("n_candidates", 200)),
        chunk_size=int(value("chunk_size", default_chunk_size)),
        unselected=_parse_unselected_policy(recipe.get("unselected")),
    )


def dump_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))
