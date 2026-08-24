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
opposite axes. We therefore prefer a verified adapter selected from
``config.json`` (``model_type`` / ``architectures``). For an unknown model, the
paired gate-up/down shapes may prove the order from their shared hidden and
intermediate dimensions. Incomplete, inconsistent, or ambiguous layouts are
refused rather than silently quantized along the wrong axis.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
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


class InferredOutInMoeLayout(MoeLayout):
    """Layout proven from paired checkpoint banks to be ``[E, out, in]``."""

    name = "inferred_out_in"
    out_in_order = True
    gate_up_fused = True


class InferredInOutMoeLayout(MoeLayout):
    """Layout proven from paired checkpoint banks to be ``[E, in, out]``."""

    name = "inferred_in_out"
    out_in_order = False
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
    "inferred_out_in": InferredOutInMoeLayout(),
    "inferred_in_out": InferredInOutMoeLayout(),
}


def layout_by_name(name: str) -> MoeLayout:
    """Resolve a layout previously recorded in a plan/manifest by its ``name``."""
    try:
        return _LAYOUTS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown MoE layout name: {name!r}") from exc


def _infer_moe_layout(
    fused_banks: Iterable[tuple[str, tuple[int, ...]]],
) -> MoeLayout:
    """Infer axis order from paired ``gate_up_proj`` and ``down_proj`` banks.

    Shape inference is safe only when both banks of the same expert module are
    present. Their shared hidden/intermediate dimensions provide an internal
    consistency check which distinguishes the two supported storage orders:

    - ``gate_up=[E, 2I, H]``, ``down=[E, H, I]`` -> ``[E, out, in]``
    - ``gate_up=[E, H, 2I]``, ``down=[E, I, H]`` -> ``[E, in, out]``

    A lone bank, malformed pair, or model whose modules disagree is refused.
    """
    banks_by_module: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    observed: list[tuple[str, tuple[int, ...]]] = []
    for name, raw_shape in fused_banks:
        shape = tuple(int(dim) for dim in raw_shape)
        observed.append((name, shape))
        if len(shape) != 3:
            continue
        module, _, proj_kind = name.rpartition(".")
        if module and proj_kind in {"gate_up_proj", "down_proj"}:
            banks_by_module[module][proj_kind] = shape

    inferred_orders: set[bool] = set()
    evidence: list[str] = []
    invalid_evidence: list[str] = []
    for module, banks in banks_by_module.items():
        gate_up = banks.get("gate_up_proj")
        down = banks.get("down_proj")
        if gate_up is None or down is None:
            invalid_evidence.append(f"{module}: gate_up={gate_up}, down={down}")
            continue

        gate_experts, gate_a, gate_b = gate_up
        down_experts, down_a, down_b = down
        out_in = (
            gate_experts == down_experts
            and gate_b == down_a
            and gate_a == 2 * down_b
        )
        in_out = (
            gate_experts == down_experts
            and gate_a == down_b
            and gate_b == 2 * down_a
        )
        if out_in == in_out:
            invalid_evidence.append(f"{module}: gate_up={gate_up}, down={down}")
            continue
        inferred_orders.add(out_in)
        evidence.append(f"{module}: gate_up={gate_up}, down={down}")

    if invalid_evidence:
        raise ValueError(
            "some fused expert modules have incomplete or inconsistent bank "
            "shapes: " + "; ".join(invalid_evidence[:8])
        )
    if inferred_orders == {True}:
        return InferredOutInMoeLayout()
    if inferred_orders == {False}:
        return InferredInOutMoeLayout()
    if len(inferred_orders) > 1:
        raise ValueError(
            "fused expert modules disagree about their 3D axis order: "
            + "; ".join(evidence[:8])
        )

    shapes = ", ".join(f"{name}={shape}" for name, shape in observed[:8])
    raise ValueError(
        "could not infer the 3D expert bank axis order from checkpoint shapes; "
        "a module must contain a consistent gate_up_proj/down_proj pair. "
        f"Observed: {shapes or 'no fused expert banks'}"
    )


def select_moe_layout(
    config: dict,
    fused_banks: Iterable[tuple[str, tuple[int, ...]]] | None = None,
) -> MoeLayout:
    """Choose a fused-expert layout adapter from a model ``config.json`` dict.

    Known model types use their verified adapter. Unknown model types may use
    checkpoint-shape inference when paired ``gate_up_proj``/``down_proj`` banks
    are supplied. Raises ``ValueError`` rather than guessing when neither path
    can establish the axis order.
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

    inference_error = None
    if fused_banks is not None:
        try:
            return _infer_moe_layout(fused_banks)
        except ValueError as exc:
            inference_error = str(exc)

    inference_detail = (
        f" Automatic shape inference also failed: {inference_error}"
        if inference_error
        else ""
    )
    raise ValueError(
        "Fused routed-expert quantization is not supported for this model: "
        f"model_type={model_type!r}, architectures={config.get('architectures')}. "
        "The 3D expert bank axis order is model specific; add a verified "
        "MoeLayout for it in flagos_compressor.core.moe_layout before quantizing."
        f"{inference_detail}"
    )
