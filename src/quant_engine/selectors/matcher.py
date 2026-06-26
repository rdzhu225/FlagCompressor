from __future__ import annotations

import re
from typing import Any

from quant_engine.core.profile import TensorInfo


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(pattern, name) or re.search(pattern, name) for pattern in patterns)


def match_selector(tensor: TensorInfo, selector: dict[str, Any]) -> bool:
    if not selector:
        return True

    include = _as_list(selector.get("include") or selector.get("pattern"))
    if include and not _matches_any(tensor.name, include):
        return False

    exclude = _as_list(selector.get("exclude"))
    if exclude and _matches_any(tensor.name, exclude):
        return False

    if "has_scale" in selector and bool(tensor.scale_name) != bool(selector["has_scale"]):
        return False

    if "dtype" in selector:
        dtypes = {str(item).removeprefix("torch.") for item in _as_list(selector["dtype"])}
        if tensor.dtype not in dtypes:
            return False

    if "element_size" in selector and tensor.element_size != int(selector["element_size"]):
        return False

    if "source_format" in selector:
        formats = set(_as_list(selector["source_format"]))
        if tensor.source_format not in formats:
            return False

    if "rank" in selector and len(tensor.shape) != int(selector["rank"]):
        return False

    return True


def match_group_selector(tensor: TensorInfo, group_config: dict[str, Any]) -> bool:
    selector = group_config.get("selector") if "selector" in group_config else group_config
    if not match_selector(tensor, selector):
        return False
    exclude_groups = group_config.get("exclude_groups") or []
    if exclude_groups:
        # Group exclusion is resolved by the planner after direct selector matching.
        return True
    return True

