"""
This module defines fixtures for illia.torch.nn.
"""

# Standard libraries
from typing import Optional, Union

# 3pps
import keras
import tensorflow as tf
import pytest

# Own modules
from illia.tf.nn.base import BayesianModule
from illia.tf.nn import Linear, Embedding, Conv2D, Conv1D
from illia.tf.distributions.base import Distribution
from illia.tf.distributions import GaussianDistribution


@pytest.fixture(
    params=[
        ((32, 30, 20, None, None)),
        (64, 20, 30, GaussianDistribution((30, 20)), GaussianDistribution((30,))),
    ]
)
def linear_fixture(request: pytest.FixtureRequest) -> tuple[Linear, tf.Tensor]:
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
    (batch_size, input_size, output_size, weights_distribution, bias_distribution) = (
        request.param
    )

    # Define model
    model: Linear = Linear(
        input_size, output_size, weights_distribution, bias_distribution
    )

    # Define inputs
    inputs: tf.Tensor = tf.random.uniform((batch_size, input_size))

    # Build the model
    model.build(inputs.shape)

    return model, inputs


@pytest.fixture(
    params=[
        (64, 100, 16, None, None, 2.0, True, False, None),
        (32, 50, 32, 0, 5.0, 1.0, False, True, GaussianDistribution((50, 32))),
    ]
)
def embedding_fixture(request: pytest.FixtureRequest) -> tuple[Embedding, tf.Tensor]:
    """
    This function is the fixture for bayesian Embedding layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Embedding instance.
        Inputs compatible with Embedding instance.
    """

    # Get parameters
    batch_size: int
    num_embeddings: int
    embeddings_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool
    weights_distribution: Optional[GaussianDistribution]
    (
        batch_size,
        num_embeddings,
        embeddings_dim,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
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
        sparse,
        weights_distribution,
    )

    # Define inputs
    inputs: tf.Tensor = tf.random.uniform(
        shape=(batch_size,), minval=0, maxval=num_embeddings, dtype=tf.int32
    )

    # Build model
    model.build(inputs.shape)

    return model, inputs


@pytest.fixture(
    params=[
        (
            32,
            3,
            9,
            3,
            1,
            "VALID",
            1,
            1,
            "NHWC",
            GaussianDistribution(shape=(3, 3, 3, 9)),
            GaussianDistribution(shape=(9,)),
            32,
            32,
        ),
        (
            64,
            6,
            6,
            [3, 3],
            [2, 1],
            "SAME",
            None,  # TODO: Error GitHub Workflow with CPU implementation dilation=[2, 1]
            1,  # TODO: Error GitHub Workflow with CPU implementation groups=2
            "NHWC",
            None,
            None,
            64,
            64,
        ),
        (
            32,
            3,
            9,
            3,
            1,
            "VALID",
            1,
            1,
            "NCHW",
            GaussianDistribution(shape=(3, 3, 3, 9)),
            GaussianDistribution(shape=(9,)),
            32,
            32,
        ),
        (
            64,
            6,
            6,
            [3, 3],
            [2, 1],
            "SAME",
            None,
            1,
            "NCHW",
            None,
            None,
            64,
            64,
        ),
    ]
)
def conv2d_fixture(request: pytest.FixtureRequest) -> tuple[Conv2D, tf.Tensor, str]:
    """
    This function is the fixture for bayesian Conv2D layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv2D instance.
        Inputs compatible with Conv2D instance.
    """

    # Get parameters
    batch_size: int
    input_channels: int
    output_channels: int
    kernel_size: Union[int, list[int]]
    stride: Union[int, list[int]]
    padding: Union[str, list[int]]
    dilation: Union[int, list[int]] = 1
    groups: int = 1
    data_format: Optional[str] = "NHWC"
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
        data_format,
        weights_distribution,
        bias_distribution,
        height,
        width,
    ) = request.param

    # Define model
    model: Conv2D = Conv2D(
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_format=data_format,
        weights_distribution=weights_distribution,
        bias_distribution=bias_distribution,
    )

    # Define inputs
    if data_format == "NHWC":
        inputs: tf.Tensor = tf.random.uniform(
            (batch_size, height, width, input_channels)
        )
    elif data_format == "NCHW":
        inputs = tf.random.uniform((batch_size, input_channels, height, width))
    else:
        raise ValueError(f"Invalid data format: {data_format}")

    # Build model
    model.build(inputs.shape)

    return model, inputs, data_format


@pytest.fixture(
    params=[
        (
            32,
            3,
            9,
            3,
            1,
            "VALID",
            1,
            1,
            "NWC",
            GaussianDistribution(shape=(3, 3, 9)),
            GaussianDistribution(shape=(9,)),
            32,
        ),
        (
            64,
            6,
            6,
            3,
            2,
            "SAME",
            None,  # TODO: Error GitHub Workflow with CPU implementation dilation=2
            1,  # TODO: Error GitHub Workflow with CPU implementation groups=2
            "NWC",
            None,
            None,
            16,
        ),
        (
            32,
            3,
            9,
            3,
            1,
            "VALID",
            1,
            1,
            "NCW",
            GaussianDistribution(shape=(3, 3, 9)),
            GaussianDistribution(shape=(9,)),
            32,
        ),
        (
            64,
            6,
            6,
            3,
            2,
            "SAME",
            None,
            1,
            "NCW",
            None,
            None,
            16,
        ),
    ]
)
def conv1d_fixture(request: pytest.FixtureRequest) -> tuple[Conv1D, tf.Tensor, str]:
    """
    This function is the fixture for bayesian Conv1D layer.

    Args:
        request: Pytest fixture request.

    Returns:
        Conv1D instance.
        Inputs compatible with Conv1D instance.
    """

    # Get parameters
    batch_size: int
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: Union[int, list[int]]
    padding: str
    dilation: Union[int, list[int]] = 1
    groups: int = 1
    data_format: Optional[str] = "NWC"
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
        data_format,
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
        data_format=data_format,
        weights_distribution=weights_distribution,
        bias_distribution=bias_distribution,
    )

    # Define inputs
    if data_format == "NWC":
        inputs: tf.Tensor = tf.random.uniform(
            (batch_size, embedding_dim, input_channels)
        )
    elif data_format == "NCW":
        inputs = tf.random.uniform((batch_size, input_channels, embedding_dim))
    else:
        raise ValueError(f"Invalid data format: {data_format}")

    # Build model
    model.build(inputs.shape)

    return model, inputs, data_format
