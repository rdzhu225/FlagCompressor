"""Layout adapters for fused 3D routed-expert banks.

A fused routed-expert bank stores every expert of one projection in a single
3D tensor. The *logical* per-expert weight is always 2D ``[out_features,
in_features]`` (compressed-tensors and vLLM's FusedMoE both expect that), but
the on-disk axis order of the bank is model specific:

- Qwen3.5-MoE stores ``[num_experts, out, in]`` directly (vLLM loads it without
  transposing). ``gate_up_proj`` fuses gate and up along ``out``.
- Qwen3-VL-MoE stores ``[num_experts, in, out]`` and vLLM transposes the last
  two axes at load time before splitting ``gate_up_proj``.

Guessing the axis order from the tensor name alone is unsafe: the two families
share the ``experts.gate_up_proj`` / ``experts.down_proj`` names but need
opposite axes. We therefore select the adapter from ``config.json``
(``model_type`` / ``architectures``) and refuse unknown layouts rather than
silently quantizing along the wrong axis.
"""

from __future__ import annotations

from dataclasses import dataclass


# Projection leaves that appear inside a fused ``experts`` bank.
FUSED_PROJECTION_LEAVES = frozenset(
    {"gate_up_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}
)


@dataclass(frozen=True)
class ExpertProjection:
    """One logical per-expert 2D projection expanded from a fused bank."""

    expert_id: int
    proj_name: str  # standard leaf, e.g. gate_proj / up_proj / down_proj
    out_features: int
    in_features: int


class MoeLayout:
    """Describes how a model's fused expert banks map to per-expert 2D weights."""

    name: str = "base"
    # True when the bank is stored as [E, out, in]; False when [E, in, out].
    out_in_order: bool = True
    # Whether ``gate_up_proj`` fuses gate and up (split into two projections).
    gate_up_fused: bool = True

    def per_expert_out_in(
        self, proj_kind: str, bank_shape: tuple[int, ...]
    ) -> tuple[int, int]:
        """Return (out_features, in_features) of ONE expert's split projection.

        ``bank_shape`` is the raw 3D storage shape. For a fused ``gate_up_proj``
        the returned ``out_features`` is per-split (gate or up), i.e. half of
        the fused output dimension.
        """
        if len(bank_shape) != 3:
            raise ValueError(
                f"fused expert bank must be 3D, got shape {bank_shape}"
            )
        _, a, b = (int(d) for d in bank_shape)
        out_dim, in_dim = (a, b) if self.out_in_order else (b, a)
        if proj_kind == "gate_up_proj":
            if not self.gate_up_fused:
                raise ValueError(
                    f"layout {self.name!r} does not fuse gate_up_proj"
                )
            out_dim = out_dim // 2
        return out_dim, in_dim

    def projection_leaves(self, proj_kind: str) -> tuple[str, ...]:
        """Standard leaf name(s) a bank projection expands into."""
        if proj_kind == "gate_up_proj":
            return ("gate_proj", "up_proj")
        return (proj_kind,)

    def expand_bank(
        self,
        proj_kind: str,
        num_experts: int,
        bank_shape: tuple[int, ...],
    ) -> list[ExpertProjection]:
        """Expand a fused bank into every per-expert logical projection."""
        out_features, in_features = self.per_expert_out_in(proj_kind, bank_shape)
        leaves = self.projection_leaves(proj_kind)
        return [
            ExpertProjection(expert_id, leaf, out_features, in_features)
            for expert_id in range(num_experts)
            for leaf in leaves
        ]


class Qwen35MoeLayout(MoeLayout):
    name = "qwen3_5_moe"
    out_in_order = True  # [E, out, in]; vLLM does NOT transpose
    gate_up_fused = True


# model_type values (and architecture fallbacks) with a verified layout.
_LAYOUTS_BY_MODEL_TYPE: dict[str, MoeLayout] = {
    "qwen3_5_moe": Qwen35MoeLayout(),
    "qwen3_5_moe_text": Qwen35MoeLayout(),
}

_LAYOUTS_BY_ARCHITECTURE: dict[str, MoeLayout] = {
    "Qwen3_5MoeForConditionalGeneration": Qwen35MoeLayout(),
    "Qwen3_5MoeForCausalLM": Qwen35MoeLayout(),
}


_LAYOUTS_BY_NAME: dict[str, MoeLayout] = {
    "qwen3_5_moe": Qwen35MoeLayout(),
}


def layout_by_name(name: str) -> MoeLayout:
    """Resolve a layout previously recorded in a plan/manifest by its ``name``."""
    try:
        return _LAYOUTS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown MoE layout name: {name!r}") from exc


def select_moe_layout(config: dict) -> MoeLayout:
    """Choose a fused-expert layout adapter from a model ``config.json`` dict.

    Raises ``ValueError`` when the model's fused-expert axis order is not known,
    so we never quantize along a guessed axis.
    """
    model_type = str(config.get("model_type", "")).strip()
    if model_type in _LAYOUTS_BY_MODEL_TYPE:
        return _LAYOUTS_BY_MODEL_TYPE[model_type]

    text_config = config.get("text_config") or {}
    text_model_type = str(text_config.get("model_type", "")).strip()
    if text_model_type in _LAYOUTS_BY_MODEL_TYPE:
        return _LAYOUTS_BY_MODEL_TYPE[text_model_type]

    for arch in config.get("architectures") or []:
        if arch in _LAYOUTS_BY_ARCHITECTURE:
            return _LAYOUTS_BY_ARCHITECTURE[arch]

    raise ValueError(
        "Fused routed-expert quantization is not supported for this model: "
        f"model_type={model_type!r}, architectures={config.get('architectures')}. "
        "The 3D expert bank axis order is model specific; add a verified "
        "MoeLayout for it in flag_compressor.core.moe_layout before quantizing."
    )
