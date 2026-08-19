import json
from argparse import Namespace

import flagos_compressor.integrations.autoround as autoround_integration
from flagos_compressor.cli.helpers import build_quantization_policy
from flagos_compressor.core.policy import (
    AutoRoundPolicy,
    CalibrationPolicy,
    QuantizationPolicy,
)
from flagos_compressor.integrations.autoround import (
    load_official_autoround_config,
    official_autoround_export_config,
    official_autoround_kwargs,
)


def test_official_autoround_config_normalizes_public_api_fields():
    config = load_official_autoround_config(
        {
            "quantization_config": {
                "scheme": "W8A16",
                "group_size": 64,
                "quant_method": "gptq",
                "provider": "auto-round",
                "iters": 321,
                "enable_quanted_input": False,
            }
        }
    )

    assert config["bits"] == 8
    assert config["group_size"] == 64
    assert config["iters"] == 321
    assert config["enable_quanted_input"] is False
    assert config["sym"] is True


def test_official_autoround_mapping_scheme_is_supported():
    config = load_official_autoround_config(
        {
            "scheme": {
                "bits": 8,
                "act_bits": 16,
                "group_size": 32,
                "sym": True,
                "data_type": "int",
            }
        }
    )

    assert config["bits"] == 8
    assert config["group_size"] == 32


def test_official_autoround_config_builds_native_policy(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "bits": 8,
                    "group_size": 64,
                    "sym": True,
                    "quant_method": "gptq",
                    "provider": "auto-round",
                    "iters": 40,
                    "batch_size": 2,
                    "gradient_accumulate_steps": 3,
                    "enable_minmax_tuning": False,
                    "enable_quanted_input": False,
                    "nsamples": 16,
                    "seqlen": 1024,
                    "seed": 7,
                }
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        recipe=None,
        select=["linear"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        method=None,
        bits=None,
        strategy=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
        autoround_config=str(config_path),
    )

    policy = build_quantization_policy(args)

    assert policy.method == "autoround"
    assert policy.format == "gptq"
    assert policy.num_bits == 8
    assert policy.group_size == 64
    assert policy.autoround == AutoRoundPolicy(
        iters=40,
        batch_size=2,
        gradient_accumulate_steps=3,
        enable_minmax_tuning=False,
        enable_quantized_input=False,
    )
    assert policy.calibration.samples == 16
    assert policy.calibration.sequence_length == 1024
    assert policy.calibration.seed == 7


def test_native_policy_translates_to_current_official_entrypoint():
    policy = QuantizationPolicy(
        selections=("linear",),
        method="autoround",
        num_bits=4,
        group_size=128,
        calibration=CalibrationPolicy(
            data=("calibration text",),
            samples=12,
            sequence_length=256,
            seed=9,
        ),
        autoround=AutoRoundPolicy(
            iters=100,
            batch_size=2,
            gradient_accumulate_steps=4,
            momentum=0.9,
            enable_quantized_input=False,
        ),
    )

    kwargs = official_autoround_kwargs(policy)
    export = official_autoround_export_config(policy)

    assert kwargs["scheme"] == "W4A16"
    assert kwargs["enable_quanted_input"] is False
    assert kwargs["lr"] == 0.01
    assert kwargs["nsamples"] == 12
    assert export["quant_method"] == "gptq"
    assert export["algorithm"] == "autoround"
    assert "dataset" not in export
    assert "scheme" not in export


def test_official_reference_builder_is_lazy_and_overrideable(monkeypatch):
    captured = {}

    def entrypoint(**kwargs):
        captured.update(kwargs)
        return "official-compressor"

    monkeypatch.setattr(
        autoround_integration,
        "official_autoround_entrypoint",
        lambda: entrypoint,
    )
    policy = QuantizationPolicy(
        selections=("linear",),
        method="autoround",
        autoround=AutoRoundPolicy(iters=3),
    )

    result = autoround_integration.build_official_autoround(
        "model",
        "tokenizer",
        policy,
        device_map="cpu",
    )

    assert result == "official-compressor"
    assert captured["model"] == "model"
    assert captured["tokenizer"] == "tokenizer"
    assert captured["scheme"] == "W4A16"
    assert captured["device_map"] == "cpu"
