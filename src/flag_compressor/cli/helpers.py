from __future__ import annotations

import json
from pathlib import Path

from flag_compressor.core.plan import ExecutionPlan
from flag_compressor.core.policy import QuantizationPolicy


def print_plan(plan: ExecutionPlan) -> None:
    print("Execution plan")
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
        raise RuntimeError("Recipe support requires PyYAML; install flag-compressor dependencies") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Quantize recipe must be a YAML mapping")
    allowed = {
        "version", "format", "method", "group_size", "n_candidates", "chunk_size",
        "select", "exclude", "other_weights",
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


def build_quantization_policy(args) -> QuantizationPolicy:
    recipe = load_quantize_recipe(args.recipe) if args.recipe else {}
    requested_format = getattr(args, "format", None) or recipe.get("format", "int4")
    if requested_format != "int4":
        raise ValueError("Only format: int4 is currently supported")

    recipe_groups, recipe_names = _parse_selector_items(recipe.get("select"))
    exclude_groups, recipe_excludes = _parse_selector_items(recipe.get("exclude"))
    selections = tuple(recipe_groups + list(args.select or ()))
    exclude_selections = tuple(exclude_groups + list(args.exclude or ()))
    include_names = tuple(recipe_names + list(args.select_name or ()))
    exclude_names = tuple(recipe_excludes + list(args.exclude_name or ()))
    if not selections and not include_names:
        raise ValueError("No INT4 weights selected; use --select/--select-name or a recipe")

    def value(name: str, default):
        cli_value = getattr(args, name)
        return cli_value if cli_value is not None else recipe.get(name, default)

    return QuantizationPolicy(
        selections=selections,
        exclude_selections=exclude_selections,
        include_names=include_names,
        exclude_names=exclude_names,
        method=value("method", "mse"),
        group_size=int(value("group_size", 32)),
        n_candidates=int(value("n_candidates", 200)),
        chunk_size=int(value("chunk_size", 4096)),
        other_weights=value("other_weights", "bf16"),
    )


def dump_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))
