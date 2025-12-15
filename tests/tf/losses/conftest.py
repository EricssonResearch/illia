"""
This module defines fixtures for illia.tf.losses.
"""

# Standard libraries
import os
from typing import Optional


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "tf"


# 3pps
import pytest
import tensorflow as tf

# Own modules
from illia.distributions import GaussianDistribution
from illia.nn import Linear


@pytest.fixture(
    params=[
        (32, 30, 20, None, None),
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
    weights_distribution: Optional[GaussianDistribution]
    bias_distribution: Optional[GaussianDistribution]
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
