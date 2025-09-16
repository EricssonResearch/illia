"""
This module contains the tests for the bayesian LSTM.
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
from illia.nn import LSTM


class TestLSTM:
    """
    This class tests the bayesian LSTM.
    """

    @pytest.mark.order(1)
    def test_init(self, lstm_fixture: tuple[LSTM, jax.Array]) -> None:
        """
        Test the LSTM constructor.
        """

        model, _ = lstm_fixture

        # Check that distributions are properly initialized
        assert hasattr(
            model, "wf_distribution"
        ), "LSTM must have forget gate weight distribution"
        assert hasattr(model, "embedding"), "LSTM must have embedding layer"
        assert model.hidden_size > 0, "Hidden size must be positive"
        assert model.output_size > 0, "Output size must be positive"

    @pytest.mark.order(2)
    def test_forward(self, lstm_fixture: tuple[LSTM, jax.Array]) -> None:
        """
        Test the LSTM forward pass.
        """

        model, inputs = lstm_fixture
        outputs, (hidden_state, cell_state) = model(inputs)

        # Check type of outputs
        assert isinstance(outputs, jax.Array), (
            f"Incorrect outputs class, expected {jax.Array} " f"and got {type(outputs)}"
        )
        assert isinstance(hidden_state, jax.Array), (
            f"Incorrect hidden_state class, expected {jax.Array} "
            f"and got {type(hidden_state)}"
        )
        assert isinstance(cell_state, jax.Array), (
            f"Incorrect cell_state class, expected {jax.Array} "
            f"and got {type(cell_state)}"
        )

        # Check output shapes
        batch_size = inputs.shape[0]
        assert (
            outputs.shape[0] == batch_size
        ), f"Expected batch size {batch_size}, got {outputs.shape[0]}"
        assert (
            outputs.shape[1] == model.output_size
        ), f"Expected output size {model.output_size}, got {outputs.shape[1]}"
        assert hidden_state.shape == (
            batch_size,
            model.hidden_size,
        ), (
            f"Expected hidden state shape {(batch_size, model.hidden_size)}, "
            f"got {hidden_state.shape}"
        )
        assert cell_state.shape == (
            batch_size,
            model.hidden_size,
        ), (
            f"Expected cell state shape {(batch_size, model.hidden_size)}, "
            f"got {cell_state.shape}"
        )

    @pytest.mark.order(3)
    def test_backward(self, lstm_fixture: tuple[LSTM, jax.Array]) -> None:
        """
        Test the LSTM backward pass.
        """

        model, inputs = lstm_fixture

        # Define loss function and compute gradients
        def loss_fn(model, inputs):
            outputs, _ = model(inputs)
            return jnp.sum(outputs)

        # TODO: Compute gradients (equivalent to backward() ??)
        _, grads = nnx.value_and_grad(loss_fn)(model, inputs)

        # Check gradients exist for all parameters
        flat_params, _ = jax.tree_util.tree_flatten(grads)

        # Check nones
        assert not any(p is None for p in flat_params), "Gradients with Nones"

    @pytest.mark.order(4)
    def test_freeze(self, lstm_fixture: tuple[LSTM, jax.Array]) -> None:
        """
        Test the freeze/unfreeze behavior of LSTM.
        """
        model, inputs = lstm_fixture

        # Compute outputs (unfrozen by default)
        outputs_first, _ = model(inputs)
        outputs_second, _ = model(inputs)

        # Check if both outputs are different (stochastic behavior)
        assert not jnp.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect outputs, different forwards are equal when at the "
            "initialization the layer should be unfrozen"
        )

        # Freeze layer
        model.freeze()

        # Compute outputs
        outputs_first, _ = model(inputs)
        outputs_second, _ = model(inputs)

        # Check if both outputs are equal (deterministic behavior)
        assert jnp.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect freezing, when layer is frozen outputs are not the same in "
            "different forward passes"
        )

        # Unfreeze layer
        model.unfreeze()

        # Compute outputs
        outputs_first, _ = model(inputs)
        outputs_second, _ = model(inputs)

        # Check if both outputs are different (stochastic behavior)
        assert not jnp.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect unfreezing, when layer is unfrozen outputs are the same in "
            "different forward passes"
        )

    @pytest.mark.order(5)
    def test_kl_cost(self, lstm_fixture: tuple[LSTM, jax.Array]) -> None:
        """
        This method is the test for the kl_cost method of LSTM layer.
        """
        model, _ = lstm_fixture

        # Compute KL cost
        kl_cost, num_params = model.kl_cost()

        # Check type of outputs
        assert isinstance(
            kl_cost, jax.Array
        ), f"Incorrect KL cost type, expected {jax.Array} and got {type(kl_cost)}"
        assert isinstance(
            num_params, int
        ), f"Incorrect num_params type, expected {int} and got {type(num_params)}"

        # Check shape of kl cost (should be scalar)
        assert (
            kl_cost.shape == ()
        ), f"Incorrect shape of KL cost, expected () and got {kl_cost.shape}"

        # Check that num_params is positive
        assert (
            num_params > 0
        ), f"Number of parameters should be positive, got {num_params}"
