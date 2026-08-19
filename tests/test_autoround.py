import torch

from flagos_compressor.quantizers.autoround import (
    AutoRoundLinear,
    SignSGD,
    fake_quantize_symmetric,
)


def test_autoround_symmetric_qdq_matches_reference_semantics():
    weight = torch.tensor([[-1.0, 0.25, 0.5, 0.75]])
    value = torch.zeros_like(weight, requires_grad=True)
    minimum = torch.ones(1, 1)
    maximum = torch.ones(1, 1)

    result = fake_quantize_symmetric(
        weight,
        bits=4,
        group_size=4,
        value=value,
        min_scale=minimum,
        max_scale=maximum,
    )

    torch.testing.assert_close(
        result.weight,
        torch.tensor([[-0.875, 0.25, 0.5, 0.75]]),
    )
    torch.testing.assert_close(result.scales, torch.tensor([[-0.125]]))
    torch.testing.assert_close(result.zeros, torch.tensor([[8.0]]))
    result.weight.sum().backward()
    assert value.grad is not None
    assert torch.count_nonzero(value.grad) > 0


def test_sign_sgd_uses_gradient_direction_and_momentum():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = SignSGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = torch.tensor([2.0, -3.0])

    optimizer.step()

    torch.testing.assert_close(parameter, torch.tensor([0.9, -0.9]))


def test_autoround_linear_keeps_native_dtype_and_shapes():
    linear = torch.nn.Linear(8, 8, bias=False)
    wrapper = AutoRoundLinear(
        linear,
        bits=8,
        group_size=4,
        enable_minmax_tuning=True,
    )

    output = wrapper(torch.randn(2, 8))
    result = wrapper.quantized()

    assert output.shape == (2, 8)
    assert result.weight.dtype == linear.weight.dtype
    assert result.scales.shape == (8, 2)
    assert result.zeros.shape == (8, 2)
