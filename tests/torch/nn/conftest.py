"""
This module defines fixtures for illia.torch.nn.
"""

# Standard libraries
from typing import Optional

# 3pps
import torch
import pytest

# Own modules
from illia.torch.nn import Linear
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
    inputs: torch.Tensor = torch.rand((output_size, input_size))

    return model, inputs
