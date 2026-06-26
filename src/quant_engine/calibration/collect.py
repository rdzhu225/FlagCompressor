from __future__ import annotations

import json
from pathlib import Path

import torch

from quant_engine.calibration.hook_collector import CalibrationCollector
from quant_engine.core.planner import assign_groups
from quant_engine.core.recipe import Recipe
from quant_engine.inspect.checkpoint_scanner import scan_hf_safetensors


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "calibrate requires transformers. Install optional dependencies with: "
            "pip install -e '.[calibration]'"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def read_jsonl_texts(path: str | Path) -> list[str]:
    texts = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
    return texts


def collect_calibration(
    recipe: Recipe,
    model_path: str,
    tokenizer_path: str | None,
    dataset_jsonl: str,
    output_dir: str,
    device: str,
) -> None:
    AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    profile = scan_hf_safetensors(recipe.model.input_path)
    groups = assign_groups(profile, recipe)
    collect_groups = recipe.calibration.collect.get("groups") or []
    layer_names = []
    for group in collect_groups:
        for tensor_name in groups.get(group, set()):
            if tensor_name.endswith(".weight"):
                layer_names.append(tensor_name[: -len(".weight")])
    layer_names = sorted(set(layer_names))
    stats = recipe.calibration.collect.get("stats") or ["hessian", "act_scales"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    collector = CalibrationCollector(layer_names, stats=stats)
    collector.register_hooks(model)

    texts = read_jsonl_texts(dataset_jsonl)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), recipe.calibration.batch_size):
            batch = texts[start : start + recipe.calibration.batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                max_length=recipe.calibration.max_length,
                truncation=True,
                padding=True,
            ).to(device)
            model(**inputs)

    collector.remove_hooks()
    collector.save(output_dir)


def run_cli(args) -> None:
    recipe = Recipe.load(args.recipe)
    model_path = args.model_path or str(recipe.model.input_path)
    dataset_jsonl = args.dataset_jsonl or recipe.calibration.dataset_jsonl
    output = args.output or recipe.calibration.output_dir
    device = args.device or recipe.backend.device or "cpu"
    if not dataset_jsonl:
        raise ValueError("Calibration dataset path must be set by recipe or --dataset-jsonl")
    if not output:
        raise ValueError("Calibration output dir must be set by recipe or --output")
    collect_calibration(
        recipe=recipe,
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_jsonl=dataset_jsonl,
        output_dir=output,
        device=device,
    )

