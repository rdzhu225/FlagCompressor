from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
from typing import Any


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str
    element_size: int
    scale_name: str | None = None
    role: str = "weight"
    source_format: str | None = None
    module_kind: str | None = None

    @property
    def module_name(self) -> str:
        if self.name.endswith(".weight"):
            return self.name[: -len(".weight")]
        return self.name


@dataclass
class ModelProfile:
    model_path: str
    format: str = "hf_safetensors"
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "format": self.format,
            "metadata": self.metadata,
            "summary": self.summary(),
            "tensors": {name: asdict(info) for name, info in self.tensors.items()},
        }

    def summary(self) -> dict[str, Any]:
        weights = [tensor for tensor in self.tensors.values() if tensor.role == "weight"]
        scaled_weights = [tensor for tensor in weights if tensor.scale_name]
        return {
            "total_tensors": len(self.tensors),
            "weight_tensors": len(weights),
            "scale_tensors": sum(1 for tensor in self.tensors.values() if tensor.role == "scale"),
            "scaled_weight_tensors": len(scaled_weights),
            "module_kinds": dict(Counter(tensor.module_kind or "unknown" for tensor in weights)),
            "source_formats": dict(Counter(tensor.source_format or "unknown" for tensor in scaled_weights)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        profile = cls(
            model_path=data["model_path"],
            format=data.get("format", "hf_safetensors"),
            metadata=data.get("metadata") or {},
        )
        profile.tensors = {
            name: TensorInfo(
                name=item["name"],
                shape=tuple(item["shape"]),
                dtype=item["dtype"],
                shard=item["shard"],
                element_size=int(item["element_size"]),
                scale_name=item.get("scale_name"),
                role=item.get("role", "weight"),
                source_format=item.get("source_format"),
                module_kind=item.get("module_kind"),
            )
            for name, item in (data.get("tensors") or {}).items()
        }
        return profile
