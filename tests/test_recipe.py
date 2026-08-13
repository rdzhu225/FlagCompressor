from argparse import Namespace
from pathlib import Path

import pytest

from flagos_compressor.cli.helpers import build_quantization_policy
from flagos_compressor.core.policy import UnselectedWeightsPolicy


def test_simple_recipe_builds_policy(tmp_path):
    recipe = tmp_path / "quantize.yaml"
    recipe.write_text(
        """
version: 1
bits: 8
method: mse
group_size: 128
unselected:
  strategy: convert
  format: bf16
select:
  - moe
  - name: '.*\\.o_proj\\.weight$'
exclude:
  - moe.shared
  - name: '.*\\.layers\\.0\\..*'
""",
        encoding="utf-8",
    )
    args = Namespace(
        recipe=str(recipe),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        method=None,
        bits=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.selections == ("moe",)
    assert policy.exclude_selections == ("moe.shared",)
    assert policy.num_bits == 8
    assert policy.group_size == 128
    assert policy.include_names == (r".*\.o_proj\.weight$",)
    assert policy.unselected == UnselectedWeightsPolicy(
        strategy="convert",
        format="bf16",
    )


def test_int8_cli_uses_w8a16_default_group_size():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.num_bits == 8
    assert policy.group_size == 128
    assert policy.chunk_size == 1024


def test_int8_channel_cli_omits_group_size():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        strategy="channel",
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.strategy == "channel"
    assert policy.group_size is None


def test_int8_channel_rejects_explicit_group_size():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        strategy="channel",
        method=None,
        group_size=128,
        n_candidates=None,
        chunk_size=None,
    )
    with pytest.raises(ValueError, match="group_size must be omitted"):
        build_quantization_policy(args)


def test_w8a8_cli_builds_dynamic_token_policy():
    args = Namespace(
        recipe=None,
        select=["moe.routed"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        activation_bits=8,
        scale_dtype="bf16",
        strategy="channel",
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.is_w8a8
    assert policy.scale_dtype == "bfloat16"
    assert policy.group_size is None


def test_w8a8_recipe_supports_bf16_scale_dtype(tmp_path):
    recipe = tmp_path / "w8a8.yaml"
    recipe.write_text(
        """
version: 1
bits: 8
activation_bits: 8
strategy: channel
scale_dtype: bf16
select:
  - attention
""",
        encoding="utf-8",
    )
    args = Namespace(
        recipe=str(recipe),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    policy = build_quantization_policy(args)

    assert policy.is_w8a8
    assert policy.scale_dtype == "bfloat16"


def test_activation_int8_requires_channel_w8():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        activation_bits=8,
        strategy="group",
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    with pytest.raises(ValueError, match="W8A8 requires"):
        build_quantization_policy(args)


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_calibrated_example_recipes_build_native_policies(method):
    args = Namespace(
        recipe=str(Path("examples/recipes") / f"{method}.yaml"),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        method=None,
        bits=None,
        strategy=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    policy = build_quantization_policy(args)

    assert policy.method == method
    assert policy.format == method
    assert policy.group_size == 128
    assert policy.calibration.samples == 128
    assert policy.calibration.sequence_length == 512
