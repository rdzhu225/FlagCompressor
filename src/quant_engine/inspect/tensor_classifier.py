from __future__ import annotations


ATTN_LINEAR_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "query",
    "key",
    "value",
    "dense",
    "c_attn",
    "c_proj",
    "qkv_proj",
    "query_key_value",
    "wq",
    "wk",
    "wv",
    "wo",
    "wq_a",
    "wq_b",
    "wkv_a",
    "wkv_b",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "q_a_proj",
    "q_b_proj",
}

MLP_LINEAR_NAMES = {
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "w1",
    "w2",
    "w3",
    "dense_h_to_4h",
    "dense_4h_to_h",
}

ATTN_PATH_HINTS = (
    ".attn.",
    ".attention.",
    ".self_attn.",
    ".self_attention.",
    ".indexer.",
)

MLP_PATH_HINTS = (
    ".mlp.",
    ".ffn.",
    ".feed_forward.",
    ".feedforward.",
)


def _weight_module_parts(name: str) -> list[str]:
    if not name.endswith(".weight"):
        return []
    return name[: -len(".weight")].split(".")


def classify_weight_module(name: str) -> str | None:
    """Classify common linear weights from checkpoint tensor names only."""
    parts = _weight_module_parts(name)
    if not parts:
        return None

    leaf = parts[-1]
    lowered = f".{name.lower()}."

    if "experts" in parts and "shared_experts" not in parts and leaf in MLP_LINEAR_NAMES:
        return "moe_mlp_linear"
    if "shared_experts" in parts and leaf in MLP_LINEAR_NAMES:
        return "shared_moe_mlp_linear"
    if leaf in MLP_LINEAR_NAMES and any(hint in lowered for hint in MLP_PATH_HINTS):
        return "mlp_linear"
    if leaf in ATTN_LINEAR_NAMES and any(hint in lowered for hint in ATTN_PATH_HINTS):
        return "attn_linear"
    if leaf in ATTN_LINEAR_NAMES:
        return "attn_linear"
    if leaf in MLP_LINEAR_NAMES:
        return "mlp_linear"
    if leaf in {"lm_head"}:
        return "lm_head"
    if leaf in {"embed_tokens", "embedding", "word_embeddings"}:
        return "embedding"
    return None


def infer_source_format(
    name: str,
    element_size: int,
    scale_name: str | None,
    module_kind: str | None = None,
) -> str | None:
    if scale_name is None:
        return None
    lower_name = name.lower()
    if element_size == 1 and (module_kind == "moe_mlp_linear" or "fp4" in lower_name):
        return "fp4_e2m1_e8m0"
    if element_size == 1:
        return "fp8_block_e8m0"
    return None
