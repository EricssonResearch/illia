"""
This module defines fixtures for illia.jax.nn.
"""

# Standard libraries
import os
from typing import Optional, Union


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "jax"


# 3pps
import jax
import pytest
from flax import nnx

# Own modules
from illia.distributions import GaussianDistribution
from illia.nn import Conv1D, Conv2D, Linear


@pytest.fixture(
    params=[
        (32, 30, 20, None, None),
        (64, 20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,))),
    ]
)
def linear_fixture(request: pytest.FixtureRequest) -> tuple[Linear, jax.Array]:
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

    # Create RNG
    rngs = nnx.Rngs(42)

    # Get parameters
    batch_size: int
    input_size: int
    output_size: int
    weights_distribution: Optional[GaussianDistribution]
    bias_distribution: Optional[GaussianDistribution]
    (
        batch_size,
        input_size,
        output_size,
        weights_distribution,
        bias_distribution,
    ) = request.param

    # Define model
    model: Linear = Linear(
        input_size=input_size,
        output_size=output_size,
        weights_distribution=weights_distribution,
        bias_distribution=bias_distribution,
        rngs=rngs,
    )

    # Define inputs
    inputs: jax.Array = jax.random.normal(rngs.params(), (batch_size, input_size))

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
def conv1d_fixture(request: pytest.FixtureRequest) -> tuple[Conv1D, jax.Array]:
    """
    This function is the fixture for bayesian Conv1D layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv1D instance.
        Inputs compatible with Conv1D instance.
    """

    # Create RNG
    rngs = nnx.Rngs(42)

    # Get parameters
    batch_size: int
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    weights_distribution: Optional[GaussianDistribution]
    bias_distribution: Optional[GaussianDistribution]
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

    # Define model
    model: Conv1D = Conv1D(
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weights_distribution=weights_distribution,
        bias_distribution=bias_distribution,
    )

    # Define inputs
    inputs: jax.Array = jax.random.normal(
        rngs.params(), (batch_size, input_channels, embedding_dim)
    )

    return model, inputs


@pytest.fixture(
    params=[
        (
            32,
            3,
            9,
            3,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            GaussianDistribution((9, 3, 3, 3)),
            GaussianDistribution((9,)),
            32,
            32,
        ),
        (64, 6, 6, (4, 4), (2, 1), (3, 1), (2, 1), 2, None, None, 64, 64),
    ]
)
def conv2d_fixture(request: pytest.FixtureRequest) -> tuple[Conv2D, jax.Array]:
    """
    This function is the fixture for bayesian Conv2D layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv2D instance.
        Inputs compatible with Conv2D instance.
    """

    # Create RNG
    rngs = nnx.Rngs(42)

    # Get parameters
    batch_size: int
    input_channels: int
    output_channels: int
    kernel_size: Union[int, tuple[int, int]]
    stride: tuple[int]
    padding: tuple[int, int]
    dilation: tuple[int]
    groups: int
    weights_distribution: Optional[GaussianDistribution]
    bias_distribution: Optional[GaussianDistribution]
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

    # Define model
    model: Conv2D = Conv2D(
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

    # Define inputs
    inputs: jax.Array = jax.random.normal(
        rngs.params(), (batch_size, input_channels, height, width)
    )

    return model, inputs
