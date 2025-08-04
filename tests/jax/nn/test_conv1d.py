"""
This module contains the tests for the bayesian Conv1D.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "jax"


# 3pps
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Own modules
from illia.nn import Conv1D


class TestConv1d:
    """
    This class tests the bayesian Conv1D.
    """

    @pytest.mark.order(1)
    def test_init(self, conv1d_fixture: tuple[Conv1D, jax.Array]) -> None:
        """
        This method is the test for the Conv1D constructor.

        Args:
            conv1d_fixture: tuple of instance of Conv1D and inputs to
                use.
        """

        model: Conv1D
        model, _ = conv1d_fixture

        # Check parameters length
        params_only = nnx.state(model, nnx.Param)
        flat_params, _ = jax.tree_util.tree_flatten(params_only)
        len_parameters = len(flat_params)
        assert (
            len_parameters == 6
        ), f"Incorrect parameters length, expected 6 and got {len_parameters}"

    @pytest.mark.order(2)
    def test_forward(self, conv1d_fixture: tuple[Conv1D, jax.Array]) -> None:
        """
        This method is the test for the Conv1D forward pass.

        Args:
            conv1d_fixture: tuple of instance of Conv1D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv1D
        inputs: jax.Array
        model, inputs = conv1d_fixture

        # Check parameters length
        outputs: jax.Array = model(inputs)

        # Check type of outputs
        assert isinstance(
            outputs, jax.Array
        ), f"Incorrect outputs class, expected {jax.Array} and got {type(outputs)}"

        # Check outputs shape
        assert outputs.shape[:2] == (inputs.shape[0], model.weights.shape[0]), (
            f"Incorrect outputs shape, expected "
            f"{(inputs.shape[0], model.weights.shape[0])} and got {outputs.shape}"
        )

    @pytest.mark.order(3)
    def test_backward(self, conv1d_fixture: tuple[Conv1D, jax.Array]) -> None:
        """
        This method is the test for the Conv1D backward pass.

        Args:
            conv1d_fixture: tuple of instance of Conv1D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv1D
        inputs: jax.Array
        model, inputs = conv1d_fixture

        # Define loss function and compute gradients
        def loss_fn(model, inputs):
            outputs = model(inputs)
            return jnp.sum(outputs)

        # TODO: Compute gradients (equivalent to backward() ??)
        _, grads = nnx.value_and_grad(loss_fn)(model, inputs)

        # Check gradients exist for all parameters
        flat_params, _ = jax.tree_util.tree_flatten(grads)

        # Check nones
        assert not any(p is None for p in flat_params), "Gradients with Nones"

    @pytest.mark.order(4)
    def test_freeze(self, conv1d_fixture: tuple[Conv1D, jax.Array]) -> None:
        """
        This method is the test for the freeze and unfreeze layers from
        Conv1D layer.

        Args:
            conv1d_fixture: tuple of instance of Conv1D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv1D
        inputs: jax.Array
        model, inputs = conv1d_fixture

        # Compute outputs
        outputs_first: jax.Array = model(inputs)
        outputs_second: jax.Array = model(inputs)

        # Check if both outputs are equal
        assert not jnp.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect outputs, different forwards are equal when at the "
            "initialization the layer should be unfrozen"
        )

        # Freeze layer
        model.freeze()

        # Compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        # Check if both outputs are equal
        assert jnp.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect freezing, when layer is frozen outputs are not the same in "
            "different forward passes"
        )

        # Unfreeze layer
        model.unfreeze()

        # Compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        # Check if both outputs are equal
        assert not jnp.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect unfreezing, when layer is unfrozen outputs are the same in "
            "different forward passes"
        )

    @pytest.mark.order(5)
    def test_kl_cost(self, conv1d_fixture: tuple[Conv1D, jax.Array]) -> None:
        """
        This method is the test for the kl_cost method of Conv1D layer.

        Args:
            conv1d_fixture: tuple of instance of Conv1D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv1D
        model, _ = conv1d_fixture

        # Compute outputs
        outputs: tuple[jax.Array, int] = model.kl_cost()

        # Check type of output
        assert isinstance(
            outputs, tuple
        ), f"Incorrect output type, expected {tuple} and got {type(outputs)}"

        # Check type of kl cost
        assert isinstance(outputs[0], jax.Array), (
            f"Incorrect output type in the first element, expected {jax.Array} and "
            f"got {type(outputs[0])}"
        )

        # Check type of num params
        assert isinstance(outputs[1], int), (
            f"Incorrect output type in the second element, expected {int} and got "
            f"{type(outputs[1])}"
        )

        # Check shape of kl cost
        assert outputs[0].shape == (), (
            f"Incorrect shape of outputs first element, expected () and got "
            f"{outputs[0].shape}"
        )
