"""
This module contains the tests for the Non-Bayesian layers Wrapper.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "jax"

# 3pps
import pytest
import jax.numpy as jnp


class TestNonParametricLayers:
    """
    This class tests the non-parametric layers integration.
    """

    @pytest.mark.order(1)
    @pytest.mark.parametrize(
        "layer_name,kwargs,input_shape",
        [
            ("MaxPool2d", {"window_shape": (2, 2)}, (32, 16, 28, 28)),
            ("AvgPool2d", {"window_shape": (2, 2)}, (32, 16, 28, 28)),
        ],
    )
    def test_pooling(self, layer_name: str, kwargs: dict, input_shape: tuple) -> None:
        """Test pooling functions."""
        pool_fn = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)
        inputs = jnp.ones(input_shape)
        output = pool_fn(inputs, **kwargs)
        assert isinstance(output, jnp.ndarray)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(2)
    @pytest.mark.parametrize("layer_name", ["ReLU", "Sigmoid", "Tanh", "GELU"])
    def test_activation(self, layer_name: str) -> None:
        """Test activation functions."""
        activation_fn = getattr(
            __import__("illia.nn", fromlist=[layer_name]), layer_name
        )
        inputs = jnp.ones((32, 16, 28, 28))
        output = activation_fn(inputs)
        assert isinstance(output, jnp.ndarray)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(3)
    @pytest.mark.parametrize("layer_name", ["BatchNorm2d", "LayerNorm", "Dropout"])
    def test_stateful_import(self, layer_name: str) -> None:
        """Test that stateful layers can be imported."""
        layer_class = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)
        assert layer_class is not None
        assert callable(layer_class)  # TODO: complex state
