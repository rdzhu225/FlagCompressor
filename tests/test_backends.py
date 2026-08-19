from flagos_compressor.backends.registry import build_backend
from flagos_compressor.cli.main import build_parser


def test_domestic_backend_is_constructed_without_hard_coded_cli_adapter():
    backend = build_backend("npu")

    assert backend.name == "npu"
    assert backend.device_name == "npu:0"
    assert backend.runtime_imports == ("torch_npu",)


def test_custom_pytorch_backend_uses_requested_device_name():
    backend = build_backend("privateuseone", "privateuseone:1")

    assert backend.name == "privateuseone"
    assert backend.device_name == "privateuseone:1"


def test_cli_accepts_domestic_pytorch_backend():
    args = build_parser().parse_args(
        [
            "quantize",
            "--input",
            "model",
            "--output",
            "output",
            "--select",
            "linear",
            "--method",
            "autoround",
            "--backend",
            "npu",
        ]
    )

    assert args.backend == "npu"
    assert args.method == "autoround"
