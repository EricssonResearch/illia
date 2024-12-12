# standard libraries
from typing import Optional

# 3pp
import pytest
import torch

# own modules
from illia.torch.nn import Linear
from illia.torch.distributions import Distribution, GaussianDistribution


@pytest.mark.parametrize(
    "input_size, output_size, weights_distribution, bias_distribution",
    [
        (30, 20, None, None),
        (20, 30, GaussianDistribution((20, 30)), GaussianDistribution((20, 30))),
    ],
)
def test_linear_init(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
) -> None:
    """
    This function is the test to the Linear constructor.

    Args:
        input_size: input size of the linear layer.
        output_size: output size of the linear layer.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        bias_distribution: distribution for the bias of the layer.
            Defaults to None.

    Returns:
        None.
    """

    # define linear layer
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # check parameters length
    len_parameters: int = len(list(model.parameters()))
    assert (
        len_parameters == 4
    ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    return None


@pytest.mark.parametrize(
    "input_size, output_size, weights_distribution, bias_distribution, batch_size",
    [
        (30, 20, None, None, 64),
        (20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,)), 32),
    ],
)
def test_linear_forward(
    input_size: int,
    output_size: int,
    weights_distribution: Optional[Distribution],
    bias_distribution: Optional[Distribution],
    batch_size: int,
) -> None:
    """
    This function is the test to the Linear constructor.

    Args:
        input_size: input size of the linear layer.
        output_size: output size of the linear layer.
        weights_distribution: distribution for the weights of the
            layer. Defaults to None.
        bias_distribution: distribution for the bias of the layer.
            Defaults to None.
        batch_size: batch size to use in the tests.

    Returns:
        None.
    """

    # define linear layer
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # get inputs
    inputs = torch.rand((batch_size, input_size))

    # check parameters length
    outputs: torch.Tensor = model(inputs)

    # check type of outputs
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect outputs class, expected {torch.Tensor} and got {type(outputs)}"

    # check outputs shape
    assert outputs.shape == (batch_size, output_size), (
        f"Incorrect outputs shape, expected {(batch_size, output_size)} and got "
        f"{outputs.shape}"
    )

    return None
