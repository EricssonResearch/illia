"""
This module defines fixtures for illia.jax.nn.
"""

# Standard libraries
import os
from typing import Optional

# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "jax"

# 3pps
import jax
from flax import nnx
import pytest

# Own modules
from illia.nn import Linear
from illia.distributions import GaussianDistribution


def linear_fixture(request: pytest.FixtureRequest) -> tuple[Linear, jax.Array]:

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
    inputs: jax.Array = jax.random.normal(
        nnx.Rngs(0).params(), (batch_size, input_size)
    )

    return model, inputs
