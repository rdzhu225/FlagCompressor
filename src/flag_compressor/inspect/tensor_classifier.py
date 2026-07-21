from __future__ import annotations


def infer_source_format(
    weight_shape: tuple[int, ...],
    element_size: int,
    scale_shape: tuple[int, ...] | None,
) -> str | None:
    """Infer the source quantization format from tensor shapes only.

    A byte-sized weight paired with a same-rows 2D scale is treated as
    MXFP4 E2M1 + E8M0 (per-row groups). A byte-sized weight paired with a
    smaller or 1D scale is treated as block-FP8 + E8M0. Anything else
    (multi-byte dtype, no scale, non-2D weight) is not a recognized
    quantized format.
    """
    if element_size != 1 or scale_shape is None:
        return None
    if len(weight_shape) != 2:
        return None
    if len(scale_shape) == 2 and scale_shape[0] == weight_shape[0]:
        return "fp4_e2m1_e8m0"
    return "fp8_block_e8m0"
