"""Import this module to register built-in format and quantization ops."""

from quant_engine.formats import fp4_e2m1 as _fp4_e2m1  # noqa: F401
from quant_engine.formats import fp8_e8m0 as _fp8_e8m0  # noqa: F401
from quant_engine.formats import int4_pack as _int4_pack  # noqa: F401
from quant_engine.quantizers import mse_int4 as _mse_int4  # noqa: F401

