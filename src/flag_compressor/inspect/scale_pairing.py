from __future__ import annotations


DEFAULT_SCALE_SUFFIXES = (
    ".scale",
    ".weight_scale_inv",
    "_scale_inv",
    ".scales",
)


def infer_weight_name_from_scale(scale_name: str) -> str | None:
    if scale_name.endswith(".weight_scale_inv"):
        return scale_name[: -len(".weight_scale_inv")] + ".weight"
    if scale_name.endswith("_scale_inv"):
        return scale_name[: -len("_scale_inv")] + ".weight"
    if scale_name.endswith(".scale"):
        prefix = scale_name[: -len(".scale")]
        if prefix.endswith(".weight"):
            return prefix
        return prefix + ".weight"
    if scale_name.endswith(".scales"):
        prefix = scale_name[: -len(".scales")]
        if prefix.endswith(".weight"):
            return prefix
        return prefix + ".weight"
    return None


def build_scale_map(weight_names: set[str]) -> dict[str, str]:
    scale_map: dict[str, str] = {}
    for name in weight_names:
        inferred = infer_weight_name_from_scale(name)
        if inferred and inferred in weight_names:
            scale_map[inferred] = name
    return scale_map

