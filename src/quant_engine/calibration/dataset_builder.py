from __future__ import annotations

import json
import random
from typing import Optional


def _require_optional_deps():
    try:
        from datasets import load_dataset
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "build-calib requires optional dependencies. Install with: "
            "pip install -e '.[calibration]'"
        ) from exc
    return load_dataset, PreTrainedTokenizerFast


def _format_squad(item: dict) -> Optional[dict]:
    texts = item.get("answers", {}).get("text", [])
    return {"question": item["question"], "answer": texts[0]} if texts else None


def _format_nq(item: dict) -> Optional[dict]:
    answers = item.get("answer", [])
    return {"question": item["question"], "answer": answers[0]} if answers else None


def _format_triviaqa(item: dict) -> Optional[dict]:
    value = item.get("answer", {}).get("value", "")
    return {"question": item["question"], "answer": value} if value else None


def _format_mmlu(item: dict) -> Optional[dict]:
    answer_idx = item["answer"]
    choices = item["choices"]
    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
        return {"question": item["question"], "answer": choices[answer_idx]}
    return None


def _format_openbookqa(item: dict) -> Optional[dict]:
    labels = item["choices"]["label"]
    texts = item["choices"]["text"]
    try:
        idx = labels.index(item["answerKey"])
    except ValueError:
        return None
    return {"question": item["question_stem"], "answer": texts[idx]}


def _format_boolq(item: dict) -> Optional[dict]:
    return {"question": item["question"], "answer": "Yes" if item["answer"] else "No"}


QA_DATASETS = {
    "squad": ("rajpurkar/squad_v2", None, "train", _format_squad),
    "nq": ("google-research-datasets/nq_open", None, "train", _format_nq),
    "triviaqa": ("trivia_qa", "rc", "train", _format_triviaqa),
    "mmlu": ("cais/mmlu", "all", "test", _format_mmlu),
    "openbookqa": ("allenai/openbookqa", "main", "test", _format_openbookqa),
    "boolq": ("google/boolq", None, "train", _format_boolq),
}


def load_and_sample(name: str, num_samples: int, seed: int) -> list[dict]:
    load_dataset, _ = _require_optional_deps()
    if name not in QA_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    hf_path, hf_config, split, format_fn = QA_DATASETS[name]
    ds = load_dataset(hf_path, hf_config, split=split, trust_remote_code=True)
    formatted = [item for row in ds if (item := format_fn(row)) is not None]
    rng = random.Random(seed)
    if len(formatted) > num_samples:
        formatted = rng.sample(formatted, num_samples)
    return formatted


def build_calibration_dataset(
    model_path: str,
    output_path: str,
    dataset_names: list[str],
    num_per_dataset: int = 64,
    tokenizer_path: str | None = None,
    seed: int = 42,
    max_length: int = 2048,
) -> None:
    _, tokenizer_cls = _require_optional_deps()
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
    samples = []
    for dataset_name in dataset_names:
        for qa in load_and_sample(dataset_name, num_per_dataset, seed):
            messages = [
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > max_length:
                text = tokenizer.decode(token_ids[:max_length], skip_special_tokens=False)
            samples.append({"text": text, "source": dataset_name})

    rng = random.Random(seed)
    rng.shuffle(samples)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} calibration samples to {output_path}")


def run_cli(args) -> None:
    build_calibration_dataset(
        model_path=args.model_path,
        output_path=args.output,
        dataset_names=args.datasets,
        num_per_dataset=args.num_per_dataset,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
        max_length=args.max_length,
    )

