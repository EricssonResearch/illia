"""
This module contains the code to test losses.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "jax"

# 3pps
import jax
import pytest
from flax import nnx

# Own modules
from illia.losses import KLDivergenceLoss
from illia.nn import Linear


class TestKLDivergenceLoss:
    """
    This class implements the tests for KLDivergenceLoss.
    """

    @pytest.mark.order(1)
    def test_forward_single(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
        """
        This method is the test for the forward pass.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        model: Linear
        model, _ = linear_fixture

        # Define loss and compute value
        loss: nnx.Module = KLDivergenceLoss()
        loss_value: jax.Array = loss(model=model)

        # Check type
        assert isinstance(loss_value, jax.Array), (
            f"Incorrect type of loss value, expected {jax.Array} and got "
            f"{type(loss_value)}"
        )

        # Check shape
        assert (
            loss_value.shape == ()
        ), f"Incorrect shape, got {loss_value.shape} and got ()"

    # @pytest.mark.order(2)
    # def test_backward_single(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
    #     """
    #     This method is the test for the backward pass.

    #     Args:
    #         linear_fixture: tuple of instance of Linear and inputs to
    #             use.
    #     """

    #     # Get model and inputs
    #     model: Linear
    #     model, _ = linear_fixture

    #     # Define loss and compute value
    #     loss: nnx.Module = KLDivergenceLoss()

    #     with tf.GradientTape() as tape:
    #         loss_value: jax.Array = loss(model=model)
    #     gradients = tape.gradient(loss_value, model.trainable_variables)

    #     # Check type of outputs
    #     for i, gradient in enumerate(gradients):
    #         # Check if parameter is none
    #         assert gradient is not None, (
    #             f"Incorrect backward computation, gradient of "
    #             f"{model.trainable_variables[i]} shouldn't be None"
    #         )
