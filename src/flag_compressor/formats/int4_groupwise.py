from __future__ import annotations

from typing import Any

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.formats.base import ArtifactResult, WeightFormat, register_weight_format


class Int4SymmetricGroupwiseFormat(WeightFormat):
    name = "int4_symmetric_groupwise"

    def from_canonical(
        self,
        tensor_name: str,
        weight: torch.Tensor,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict[str, Any],
    ) -> ArtifactResult:
        quantizer = params.get("quantizer", "mse")
        if quantizer != "mse":
            raise NotImplementedError(f"Unsupported INT4 quantizer: {quantizer}")
        int4_values, scales = backend.run(
            "mse_int4_quant",
            weight,
            group_size=int(params.get("group_size", 32)),
            n_candidates=int(params.get("n_candidates", 200)),
            chunk_size=int(params.get("chunk_size", 4096)),
            context=context,
        )
        packed = backend.run("int4_pack", int4_values, context=context)
        scale_name = tensor_name + params.get("scale_suffix", ".scale")
        return ArtifactResult(
            tensors={tensor_name: packed.cpu(), scale_name: scales.cpu()},
            generated_tensor_names=(scale_name,),
        )


register_weight_format(Int4SymmetricGroupwiseFormat())
