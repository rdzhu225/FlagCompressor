from argparse import Namespace

from flag_compressor.cli.helpers import build_quantization_policy


def test_simple_recipe_builds_policy(tmp_path):
    recipe = tmp_path / "quantize.yaml"
    recipe.write_text(
        """
version: 1
format: int4
method: mse
group_size: 64
select:
  - moe
  - name: '.*\\.o_proj\\.weight$'
exclude:
  - moe.shared
  - name: '.*\\.layers\\.0\\..*'
other_weights: bf16
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
        group_size=None,
        n_candidates=None,
        chunk_size=None,
        other_weights=None,
    )
    policy = build_quantization_policy(args)
    assert policy.selections == ("moe",)
    assert policy.exclude_selections == ("moe.shared",)
    assert policy.group_size == 64
    assert policy.include_names == (r".*\.o_proj\.weight$",)
