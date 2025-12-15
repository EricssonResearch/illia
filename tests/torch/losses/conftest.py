"""
This module defines fixtures for illia.torch.losses.
"""

# Standard libraries
import os
from typing import Optional


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "torch"


# 3pps
import pytest
import torch

# Own modules
from illia.distributions import GaussianDistribution
from illia.nn import BayesianModule, Linear
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
