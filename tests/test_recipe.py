from argparse import Namespace

from flagos_compressor.cli.helpers import build_quantization_policy
from flagos_compressor.core.policy import UnselectedWeightsPolicy


def test_simple_recipe_builds_policy(tmp_path):
    recipe = tmp_path / "quantize.yaml"
    recipe.write_text(
        """
version: 1
method: mse
group_size: 64
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
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.selections == ("moe",)
    assert policy.exclude_selections == ("moe.shared",)
    assert policy.group_size == 64
    assert policy.include_names == (r".*\.o_proj\.weight$",)
    assert policy.unselected == UnselectedWeightsPolicy(
        strategy="convert",
        format="bf16",
    )
