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
from illia.nn import LSTM, Conv1d, Conv2d, Embedding, Linear


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
def conv1d_fixture(request: pytest.FixtureRequest) -> tuple[Conv1d, jax.Array]:
    """
    This function is the fixture for bayesian Conv1d layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv1d instance.
        Inputs compatible with Conv1d instance.
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
    model: Conv1d = Conv1d(
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
def conv2d_fixture(request: pytest.FixtureRequest) -> tuple[Conv2d, jax.Array]:
    """
    This function is the fixture for bayesian Conv2d layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv2d instance.
        Inputs compatible with Conv2d instance.
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

    # Define inputs
    inputs: jax.Array = jax.random.normal(
        rngs.params(), (batch_size, input_channels, height, width)
    )

    return model, inputs


@pytest.fixture(
    params=[
        (64, 100, 16, None, None, 2.0, False, None),
        (32, 50, 32, 0, 5.0, 1.0, False, GaussianDistribution((50, 32))),
    ]
)
def embedding_fixture(request: pytest.FixtureRequest) -> tuple[Embedding, jax.Array]:
    """
    This function is the fixture for bayesian Embedding layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Embedding instance.
        Inputs compatible with Embedding instance.
    """

    # Create RNG
    rngs = nnx.Rngs(42)

    # Get parameters
    batch_size: int
    num_embeddings: int
    embeddings_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weights_distribution: Optional[GaussianDistribution]
    (
        batch_size,
        num_embeddings,
        embeddings_dim,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        weights_distribution,
    ) = request.param

    # Define model
    model: Embedding = Embedding(
        num_embeddings,
        embeddings_dim,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        weights_distribution,
        rngs,
    )

    # Define inputs
    inputs: jax.Array = jax.random.randint(
        key=jax.random.PRNGKey(42),
        shape=(batch_size,),
        minval=0,
        maxval=num_embeddings,
    )

    return model, inputs


@pytest.fixture(
    params=[
        (32, 128, 15, 20, 30, 10, None, None, 2.0, False),
    ]
)
def lstm_fixture(request: pytest.FixtureRequest) -> tuple[LSTM, jax.Array]:
    """
    Fixture for the Bayesian LSTM layer.

    Args:
        request: Pytest fixture request with parameters:
            batch_size, seq_len, num_embeddings, embeddings_dim,
            hidden_size, output_size, padding_idx, max_norm, norm_type,
            scale_grad_by_freq, sparse.

    Returns:
        LSTM instance and a random input tensor with token indices.
    """

    # Create RNG
    rngs = nnx.Rngs(42)

    (
        batch_size,
        seq_len,
        num_embeddings,
        embeddings_dim,
        hidden_size,
        output_size,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
    ) = request.param

    # Define model
    model: LSTM = LSTM(
        num_embeddings=num_embeddings,
        embeddings_dim=embeddings_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        rngs=rngs,
    )

    # Define inputs
    inputs: jax.Array = jax.random.randint(
        key=jax.random.PRNGKey(42),
        shape=(batch_size, seq_len, 1),
        minval=0,
        maxval=num_embeddings,
    )

    return model, inputs
