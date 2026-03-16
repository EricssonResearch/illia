"""
This module contains the tests for the Non-Bayesian layers Wrapper.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "jax"

# 3pps
import jax.numpy as jnp
import pytest


class TestNonParametricLayers:
    """
    This class tests the non-parametric layers integration.
    """

    @pytest.mark.order(1)
    @pytest.mark.parametrize(
        "layer_name,input_shape,window_shape",
        [
            ("MaxPool1d", (32, 16, 28), (2,)),
            ("MaxPool2d", (32, 16, 28, 28), (2, 2)),
            ("AvgPool1d", (32, 16, 28), (2,)),
            ("AvgPool2d", (32, 16, 28, 28), (2, 2)),
        ],
    )
    def test_pooling(
        self, layer_name: str, input_shape: tuple, window_shape: tuple
    ) -> None:
        """
        Test pooling functions.

        Args:
            layer_name: Name of the pooling function to test.
            input_shape: Shape of the input tensor for testing.
            window_shape: Shape of the pooling window.
        """
        pool_fn = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)()
        inputs = jnp.ones(input_shape)
        output = pool_fn(inputs, window_shape=window_shape)
        assert isinstance(output, jnp.ndarray)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(2)
    @pytest.mark.parametrize("layer_name", ["ReLU", "Sigmoid", "Tanh", "GELU"])
    def test_activation(self, layer_name: str) -> None:
        """
        Test activation functions.

        Args:
            layer_name: Name of the activation function to test.
        """
        activation_fn = getattr(
            __import__("illia.nn", fromlist=[layer_name]), layer_name
        )()
        inputs = jnp.ones((32, 16, 28, 28))
        output = activation_fn(inputs)
        assert isinstance(output, jnp.ndarray)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(3)
    def test_regularization(self, rngs_fixture) -> None:
        """
        Test regularization layers.

        Args:
            rngs_fixture: JAX RNG fixture for consistent random number generation.
        """
        Dropout = getattr(__import__("illia.nn", fromlist=["Dropout"]), "Dropout")
        layer_instance = Dropout(rate=0.5, rngs=rngs_fixture)
        assert layer_instance is not None
        assert callable(layer_instance)

    @pytest.mark.order(4)
    def test_normalization(self, rngs_fixture) -> None:
        """
        Test normalization layers.

        Args:
            rngs_fixture: JAX RNG fixture for consistent random number generation.
        """
        # Test BatchNorm1d
        BatchNorm1d = getattr(
            __import__("illia.nn", fromlist=["BatchNorm1d"]), "BatchNorm1d"
        )
        layer1d = BatchNorm1d(num_features=16, rngs=rngs_fixture)
        assert layer1d is not None
        assert callable(layer1d)

        # Test BatchNorm2d
        BatchNorm2d = getattr(
            __import__("illia.nn", fromlist=["BatchNorm2d"]), "BatchNorm2d"
        )
        layer2d = BatchNorm2d(num_features=16, rngs=rngs_fixture)
        assert layer2d is not None
        assert callable(layer2d)

        # Test LayerNorm
        LayerNorm = getattr(__import__("illia.nn", fromlist=["LayerNorm"]), "LayerNorm")
        layer_norm = LayerNorm(num_features=16, rngs=rngs_fixture)
        assert layer_norm is not None
        assert callable(layer_norm)
