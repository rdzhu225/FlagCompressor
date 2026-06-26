from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
            "tensors": {name: asdict(info) for name, info in self.tensors.items()},
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
            )
            for name, item in (data.get("tensors") or {}).items()
        }
        return profile

