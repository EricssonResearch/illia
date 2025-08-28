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

    @pytest.mark.order(2)
    def test_backward_single(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
        """
        This method is the test for the backward pass.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        model: Linear
        model, _ = linear_fixture

        # Define loss and compute value
        loss: nnx.Module = KLDivergenceLoss()

        # Define loss function and compute gradients
        def loss_fn(loss, model):
            return loss(model)

        # TODO: Compute gradients (equivalent to backward() ??)
        _, grads = nnx.value_and_grad(loss_fn)(loss, model)

        # Check gradients exist for all parameters
        flat_params, _ = jax.tree_util.tree_flatten(grads)

        # Check nones
        assert not any(p is None for p in flat_params), "Gradients with Nones"
