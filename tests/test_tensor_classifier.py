from flag_compressor.inspect.tensor_classifier import infer_storage_format


def test_no_scale_returns_none():
    assert infer_storage_format((16, 32), element_size=2, scale_shape=None) is None
    assert infer_storage_format((16, 32), element_size=1, scale_shape=None) is None


def test_multibyte_weight_returns_none():
    assert infer_storage_format((16, 32), element_size=2, scale_shape=(16, 1)) is None


def test_byte_weight_with_per_row_2d_scale_is_fp4():
    # MXFP4 typical layout: same number of rows as weight, one scale per group along cols.
    assert (
        infer_storage_format((128, 16), element_size=1, scale_shape=(128, 1))
        == "fp4_e2m1_e8m0"
    )


def test_rowwise_int8_layout_is_not_misclassified_as_fp4():
    assert infer_storage_format((128, 32), element_size=1, scale_shape=(128, 1)) is None


def test_byte_weight_with_block_scale_is_fp8():
    # Block-FP8 typical layout: a scale block per 128x128 tile.
    assert (
        infer_storage_format((256, 512), element_size=1, scale_shape=(2, 4))
        == "fp8_block_e8m0"
    )
    assert (
        infer_storage_format((256, 512), element_size=1, scale_shape=(8,))
        == "fp8_block_e8m0"
    )
