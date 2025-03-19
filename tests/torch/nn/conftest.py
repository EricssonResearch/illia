"""
This module defines fixtures for illia.torch.nn.
"""

# Standard libraries
from typing import Optional, Union

# 3pps
import torch
import pytest

# Own modules
from illia.torch.nn import Linear, Conv1d, Conv2d
from illia.torch.distributions import Distribution, GaussianDistribution


@pytest.fixture(
    params=[
        ((32, 30, 20, None, None)),
        (64, 20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,))),
    ]
)
def linear_fixture(request: pytest.FixtureRequest) -> tuple[Linear, torch.Tensor]:
    """
    This function is the fixture for bayesian Linear layer.

    Args:
        request: Pytest fixture request with the following fields:
            batch_size, input_size, output_size, weights_distribution,
            bias_distribution.

    Returns:
        Linear instance.
        Inputs compatible with Linear instance.
    """

    # Get parameters
    batch_size: int
    input_size: int
    output_size: int
    weights_distribution: Optional[Distribution]
    bias_distribution: Optional[Distribution]
    (
        batch_size,
        input_size,
        output_size,
        weights_distribution,
        bias_distribution,
    ) = request.param

    # Define model and inputs
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )
    inputs: torch.Tensor = torch.rand((batch_size, input_size))

    return model, inputs


@pytest.fixture(
    params=[
        (
            32,
            3,
            9,
            3,
            1,
            0,
            1,
            1,
            GaussianDistribution((9, 3, 3, 3)),
            GaussianDistribution((9,)),
            32,
            32,
        ),
        (64, 6, 6, (4, 4), (2, 1), (3, 1), (2, 1), 2, None, None, 64, 64),
    ]
)
def conv2d_fixture(request: pytest.FixtureRequest) -> tuple[Conv2d, torch.Tensor]:
    """
    This function is the fixture for bayesian Conv2d layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv2d instance.
        Inputs compatible with Conv2d instance.
    """

    # Get parameters
    batch_size: int
    input_channels: int
    output_channels: int
    kernel_size: Union[int, tuple[int, int]]
    stride: Union[int, tuple[int, int]]
    padding: Union[int, tuple[int, int]]
    dilation: Union[int, tuple[int, int]] = 1
    groups: int = 1
    weights_distribution: Optional[Distribution]
    bias_distribution: Optional[Distribution]
    height: int
    width: int
    (
        batch_size,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
        height,
        width,
    ) = request.param

    # Define model and inputs
    model: Conv2d = Conv2d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
    )
    inputs: torch.Tensor = torch.rand((batch_size, input_channels, height, width))

    return model, inputs


@pytest.fixture(
    params=[
        (
            32,
            3,
            9,
            3,
            1,
            0,
            1,
            1,
            GaussianDistribution((9, 3, 3)),
            GaussianDistribution((9,)),
            32,
        ),
        (64, 6, 6, 4, 2, 3, 2, 2, None, None, 16),
    ]
)
def conv1d_fixture(request: pytest.FixtureRequest) -> tuple[Conv1d, torch.Tensor]:
    """
    This function is the fixture for bayesian Conv1d layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv1d instance.
        Inputs compatible with Conv1d instance.
    """

    # Get parameters
    batch_size: int
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    weights_distribution: Optional[Distribution]
    bias_distribution: Optional[Distribution]
    embedding_dim: int
    (
        batch_size,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
        embedding_dim,
    ) = request.param

    # Define model and inputs
    model: Conv1d = Conv1d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weights_distribution,
        bias_distribution,
    )
    inputs: torch.Tensor = torch.rand((batch_size, input_channels, embedding_dim))

    return model, inputs
