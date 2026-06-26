from __future__ import annotations

from quant_engine.core.plan import ExecutionPlan
from quant_engine.core.profile import ModelProfile, TensorInfo
from quant_engine.core.recipe import Recipe, RuleConfig
from quant_engine.selectors.matcher import match_group_selector, match_selector


def _when_matches(tensor: TensorInfo, rule: RuleConfig) -> bool:
    if not rule.when:
        return True
    source_format = rule.when.get("source_format")
    if source_format is not None:
        allowed = source_format if isinstance(source_format, list) else [source_format]
        if tensor.source_format not in allowed:
            return False
    dtype = rule.when.get("dtype")
    if dtype is not None:
        allowed = dtype if isinstance(dtype, list) else [dtype]
        allowed = [str(item).removeprefix("torch.") for item in allowed]
        if tensor.dtype not in allowed:
            return False
    has_scale = rule.when.get("has_scale")
    if has_scale is not None and bool(tensor.scale_name) != bool(has_scale):
        return False
    return True


def assign_groups(profile: ModelProfile, recipe: Recipe) -> dict[str, set[str]]:
    direct_matches: dict[str, set[str]] = {}
    for group_name, config in recipe.module_groups.items():
        direct_matches[group_name] = {
            tensor.name
            for tensor in profile.tensors.values()
            if tensor.role != "scale" and match_group_selector(tensor, config)
        }

    resolved: dict[str, set[str]] = {}
    for group_name, config in recipe.module_groups.items():
        names = set(direct_matches[group_name])
        for excluded in config.get("exclude_groups") or []:
            names -= direct_matches.get(excluded, set())
        resolved[group_name] = names
    return resolved


def build_plan(profile: ModelProfile, recipe: Recipe) -> ExecutionPlan:
    groups = assign_groups(profile, recipe)
    plan = ExecutionPlan()
    matched_names: set[str] = set()

    tensors = [tensor for tensor in profile.tensors.values() if tensor.role != "scale"]
    for tensor in tensors:
        selected_rule: RuleConfig | None = None
        selected_group: str | None = None
        for rule in recipe.rules:
            in_group = True
            if rule.group:
                in_group = tensor.name in groups.get(rule.group, set())
            selector_ok = match_selector(tensor, rule.selector)
            if in_group and selector_ok and _when_matches(tensor, rule):
                selected_rule = rule
                selected_group = rule.group
                break
        if selected_rule:
            plan.add_action(tensor, selected_rule, selected_group)
            matched_names.add(tensor.name)
        else:
            plan.kept_tensors.append(tensor)
            if tensor.element_size == 1 and tensor.scale_name:
                plan.unmatched_quantized_tensors.append(tensor)

    for group_name, names in groups.items():
        plan.group_counts.setdefault(group_name, len(names))
    return plan

