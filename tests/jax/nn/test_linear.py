"""
This module contains the code to test the bayesian Linear layer.
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
from illia.nn import Linear


class TestLinear:
    """
    This class tests the Linear bayesian layer.
    """

    @pytest.mark.order(1)
    def test_init(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
        """
        This method is the test for the Linear constructor.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        model: Linear
        model, _ = linear_fixture

        # Check parameters length
        params_only = nnx.state(model, nnx.Param)
        flat_params, _ = jax.tree_util.tree_flatten(params_only)
        len_parameters = len(flat_params)
        assert (
            len_parameters == 6
        ), f"Incorrect parameters length, expected 6 and got {len_parameters}"

    @pytest.mark.order(2)
    def test_forward(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
        """
        This method is the test for the Linear forward pass.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        model: Linear
        inputs: jax.Array
        model, inputs = linear_fixture

        # Check parameters length
        outputs: jax.Array = model(inputs)

        # Check type of outputs
        assert isinstance(
            outputs, jax.Array
        ), f"Incorrect outputs class, expected {jax.Array} and got {type(outputs)}"

        # Check outputs shape
        assert outputs.shape == (inputs.shape[0], model.weights.shape[0]), (
            f"Incorrect outputs shape, expected "
            f"{(inputs.shape[0], model.weights.shape[0])} and got {outputs.shape}"
        )

    # @pytest.mark.order(3)
    # def test_backward(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
    #     """
    #     This method is the test for the Linear backward pass.

    #     Args:
    #         linear_fixture: tuple of instance of Linear and inputs to
    #             use.
    #     """

    #     # Get model and inputs
    #     model: Linear
    #     inputs: jax.Array
    #     model, inputs = linear_fixture

    #     # check parameters length
    #     outputs: jax.Array = model(inputs)
    #     outputs.sum().backward()

    #     # Check type of outputs
    #     for name, parameter in model.named_parameters():
    #         # check if parameter is none
    #         assert parameter.grad is not None, (
    #             f"Incorrect backward computation, gradient of {name} shouldn't be "
    #             f"None"
    #         )

    @pytest.mark.order(4)
    def test_freeze(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
        """
        This method is the test for the freeze and unfreeze layers from
        Linear layer.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        model: Linear
        inputs: jax.Array
        model, inputs = linear_fixture

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
    def test_kl_cost(self, linear_fixture: tuple[Linear, jax.Array]) -> None:
        """
        This method is the test for the kl_cost method of Linear layer.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        model: Linear
        model, _ = linear_fixture

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
