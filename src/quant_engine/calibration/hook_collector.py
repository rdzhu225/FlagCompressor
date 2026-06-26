from __future__ import annotations

import json
from pathlib import Path

import torch


class CalibrationCollector:
    """Collect Hessian and activation scale statistics from selected modules."""

    def __init__(self, layer_names: list[str], stats: list[str] | None = None) -> None:
        self.layer_names = set(layer_names)
        self.stats_to_collect = set(stats or ["hessian", "act_scales"])
        self.hooks = []
        self.stats: dict[str, dict[str, torch.Tensor]] = {}
        self.n_samples: dict[str, int] = {}

    def _hook_fn(self, name: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.float()
            tokens = x.shape[0]
            if name not in self.stats:
                features = x.shape[1]
                self.stats[name] = {}
                if "hessian" in self.stats_to_collect:
                    self.stats[name]["hessian"] = torch.zeros(features, features, device=x.device)
                if "act_scales" in self.stats_to_collect:
                    self.stats[name]["act_scales"] = torch.zeros(features, device=x.device)
                self.n_samples[name] = 0
            if "hessian" in self.stats[name]:
                self.stats[name]["hessian"].add_(x.T @ x)
            if "act_scales" in self.stats[name]:
                self.stats[name]["act_scales"].add_(x.abs().sum(dim=0))
            self.n_samples[name] += tokens

        return hook

    def register_hooks(self, model: torch.nn.Module) -> None:
        module_names = {name for name, _ in model.named_modules()}
        for name, module in model.named_modules():
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
        missing = self.layer_names - module_names
        if missing:
            print(f"Warning: {len(missing)} calibration layers were not found in the model.")
        print(f"Registered {len(self.hooks)} calibration hooks.")

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def finalize(self) -> dict[str, dict[str, torch.Tensor]]:
        result = {}
        for name, stat in self.stats.items():
            n = self.n_samples.get(name, 0)
            if n <= 0:
                result[name] = stat
                continue
            result[name] = {key: value / n for key, value in stat.items()}
        return result

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        finalized = self.finalize()
        index = {}
        for name, stat in finalized.items():
            filename = name.replace(".", "_").replace("/", "_") + ".pt"
            torch.save(
                {
                    **{key: value.cpu() for key, value in stat.items()},
                    "n_samples": self.n_samples.get(name, 0),
                },
                output / filename,
            )
            index[name] = filename
        with (output / "calibration_index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        print(f"Saved calibration stats for {len(index)} layers to {output}")

    @staticmethod
    def load(input_dir: str | Path) -> dict[str, dict[str, torch.Tensor]]:
        base = Path(input_dir)
        with (base / "calibration_index.json").open("r", encoding="utf-8") as f:
            index = json.load(f)
        result = {}
        for name, filename in index.items():
            data = torch.load(base / filename, map_location="cpu", weights_only=True)
            result[name] = {
                key: value
                for key, value in data.items()
                if key in {"hessian", "act_scales"}
            }
        return result

