"""
This module defines fixtures for illia.torch.nn.
"""

# Standard libraries
import os
from typing import Optional, Union


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "torch"


# 3pps
import pytest
import torch

# Own modules
from illia.distributions import GaussianDistribution
from illia.nn import LSTM, BayesianModule, Conv1D, Conv2D, Embedding, Linear
from tests.torch.nn.utils import BayesianComposedModel, ComposedModel


@pytest.fixture(
    params=[
        (32, 30, 20, None, None),
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
        input_size, output_size, weights_distribution, bias_distribution
    )

    # Define inputs
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
def conv2d_fixture(request: pytest.FixtureRequest) -> tuple[Conv2D, torch.Tensor]:
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
    kernel_size: Union[int, tuple[int, int]]
    stride: Union[int, tuple[int, int]]
    padding: Union[int, tuple[int, int]]
    dilation: Union[int, tuple[int, int]] = 1
    groups: int = 1
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
def conv1d_fixture(request: pytest.FixtureRequest) -> tuple[Conv1D, torch.Tensor]:
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
    inputs: torch.Tensor = torch.rand((batch_size, input_channels, embedding_dim))

    return model, inputs


@pytest.fixture(
    params=[
        (64, 100, 16, None, None, 2.0, True, False, None),
        (32, 50, 32, 0, 5.0, 1.0, False, True, GaussianDistribution((50, 32))),
    ]
)
def embedding_fixture(request: pytest.FixtureRequest) -> tuple[Embedding, torch.Tensor]:
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
    inputs: torch.Tensor = torch.randint(0, num_embeddings, (batch_size,))

    return model, inputs


@pytest.fixture(params=[2, 3, 4])
def composed_fixture(
    request: pytest.FixtureRequest, linear_fixture: tuple[Linear, torch.Tensor]
) -> tuple[BayesianModule, torch.nn.Module, torch.Tensor]:
    """
    This fixture creates a module with several bayesian Linear modules.

    Args:
        request: Pytest fixture request.
        linear_fixture: Linear fixture with Linear model and inputs.

    Returns:
        Composed model with several Linear layers.
        Inputs tensor.
    """

    # Get fixture parameters
    num_models: int = request.param

    # Get model and inputs
    model: Linear
    inputs: torch.Tensor
    model, inputs = linear_fixture

    # Define composed model
    composed_model: torch.nn.Module = ComposedModel(model, num_models)
    bayesian_composed_model: BayesianModule = BayesianComposedModel(model, num_models)

    return bayesian_composed_model, composed_model, inputs


@pytest.fixture(
    params=[
        (32, 128, 15, 20, 30, 10, None, None, 2.0, False, False),
    ]
)
def lstm_fixture(request: pytest.FixtureRequest) -> tuple[LSTM, torch.Tensor]:
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
        sparse,
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
        sparse=sparse,
    )

    # Create integer token indices for embedding layer
    # Shape: (batch_size, seq_len, 1)
    inputs: torch.Tensor = torch.randint(
        0, num_embeddings, (batch_size, seq_len, 1), dtype=torch.long
    )

    return model, inputs
