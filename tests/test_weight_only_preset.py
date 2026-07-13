from quant_engine.core.recipe import Recipe
from quant_engine.presets.weight_only import fp4_moe_int4_fp8_linear_bf16_recipe


def test_weight_only_preset_is_loadable_recipe():
    data = fp4_moe_int4_fp8_linear_bf16_recipe(
        "/models/in",
        "/models/out",
        "profile.json",
    )
    recipe = Recipe.from_dict(data)

    assert recipe.model.input_path.as_posix() == "/models/in"
    assert recipe.model.output_path.as_posix() == "/models/out"
    assert recipe.discover["profile"] == "profile.json"
    assert [rule.transform for rule in recipe.rules] == ["fp4_to_int4", "fp8_to_bf16"]

