from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    input_path: Path
    output_path: Path | None = None
    format: str = "hf_safetensors"


@dataclass(frozen=True)
class BackendConfig:
    name: str = "cpu"
    device: str | None = None
    fallback_policy: str = "warn"
    op_placement: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleConfig:
    name: str
    transform: str
    group: str | None = None
    selector: dict[str, Any] = field(default_factory=dict)
    when: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    quantizer: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = False
    dataset_jsonl: str | None = None
    output_dir: str | None = None
    max_length: int = 2048
    batch_size: int = 1
    collect: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recipe:
    version: int
    model: ModelConfig
    backend: BackendConfig
    module_groups: dict[str, dict[str, Any]]
    rules: list[RuleConfig]
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    discover: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        recipe_path = Path(path)
        with recipe_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        model_data = data.get("model") or {}
        if "input_path" not in model_data:
            raise ValueError("Recipe missing model.input_path")
        model = ModelConfig(
            input_path=Path(model_data["input_path"]).expanduser(),
            output_path=Path(model_data["output_path"]).expanduser()
            if model_data.get("output_path")
            else None,
            format=model_data.get("format", "hf_safetensors"),
        )

        backend_data = data.get("backend") or {}
        backend = BackendConfig(
            name=backend_data.get("name", "cpu"),
            device=backend_data.get("device"),
            fallback_policy=backend_data.get("fallback_policy", "warn"),
            op_placement=backend_data.get("op_placement", data.get("op_placement") or {}),
        )

        rules = [
            RuleConfig(
                name=item["name"],
                group=item.get("group"),
                selector=item.get("selector") or {},
                when=item.get("when") or {},
                transform=item["transform"],
                params=item.get("params") or {},
                quantizer=item.get("quantizer") or {},
                output=item.get("output") or {},
            )
            for item in data.get("rules", [])
        ]
        if not rules:
            raise ValueError("Recipe must define at least one rule")

        calib_data = data.get("calibration") or {}
        calibration = CalibrationConfig(
            enabled=bool(calib_data.get("enabled", False)),
            dataset_jsonl=calib_data.get("dataset_jsonl"),
            output_dir=calib_data.get("output_dir"),
            max_length=int(calib_data.get("max_length", 2048)),
            batch_size=int(calib_data.get("batch_size", 1)),
            collect=calib_data.get("collect") or {},
        )

        return cls(
            version=int(data.get("version", 1)),
            model=model,
            backend=backend,
            module_groups=data.get("module_groups") or {},
            rules=rules,
            calibration=calibration,
            discover=data.get("discover") or {},
            raw=data,
        )

