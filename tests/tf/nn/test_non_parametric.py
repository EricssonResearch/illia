"""
This module contains the tests for the Non-Bayesian layers Wrapper.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "tf"

# 3pps
import pytest
import tensorflow as tf


tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


class TestNonParametricLayers:
    """
    This class tests the non-parametric layers integration.
    """

    @pytest.mark.order(1)
    @pytest.mark.parametrize(
        "layer_name,kwargs,input_shape,expected_shape",
        [
            ("MaxPool2d", {"pool_size": 2}, (32, 28, 28, 16), (32, 14, 14, 16)),
            ("AvgPool2d", {"pool_size": 2}, (32, 28, 28, 16), (32, 14, 14, 16)),
            ("MaxPool1d", {"pool_size": 2}, (32, 64, 16), (32, 32, 16)),
            ("AvgPool1d", {"pool_size": 2}, (32, 64, 16), (32, 32, 16)),
            ("AdaptiveAvgPool2d", {}, (32, 28, 28, 16), (32, 16)),
        ],
    )
    def test_pooling(
        self, layer_name: str, kwargs: dict, input_shape: tuple, expected_shape: tuple
    ) -> None:
        """Test pooling layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)(
            **kwargs
        )
        inputs = tf.random.uniform(input_shape)
        output = layer(inputs)
        assert isinstance(output, tf.Tensor)
        assert tuple(output.shape) == expected_shape
        assert output.dtype == inputs.dtype

    @pytest.mark.order(2)
    @pytest.mark.parametrize(
        "layer_name", ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "GELU"]
    )
    def test_activation(self, layer_name: str) -> None:
        """Test activation layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)()
        inputs = tf.random.uniform((32, 28, 28, 16))
        output = layer(inputs)
        assert isinstance(output, tf.Tensor)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(3)
    @pytest.mark.parametrize(
        "layer_name,input_shape",
        [
            ("BatchNorm1d", (32, 64, 16)),
            ("BatchNorm2d", (32, 28, 28, 16)),
            ("LayerNorm", (32, 28, 28, 16)),
        ],
    )
    def test_normalization(self, layer_name: str, input_shape: tuple) -> None:
        """Test normalization layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)()
        inputs = tf.random.normal(input_shape, mean=5.0, stddev=2.0)
        output = layer(inputs, training=True)
        assert isinstance(output, tf.Tensor)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype
        # assert tf.abs(tf.reduce_mean(output)) < 0.1
        # assert tf.abs(tf.math.reduce_std(output) - 1.0) < 0.1

    @pytest.mark.order(4)
    @pytest.mark.parametrize(
        "layer_name,rate,input_shape",
        [
            ("Dropout", 0.5, (32, 28, 28, 16)),
            ("Dropout2d", 0.5, (32, 28, 28, 16)),
        ],
    )
    def test_regularization(
        self, layer_name: str, rate: float, input_shape: tuple
    ) -> None:
        """Test regularization layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)(rate)
        inputs = tf.random.uniform(input_shape)
        output = layer(inputs, training=True)
        assert isinstance(output, tf.Tensor)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(5)
    def test_flatten(self) -> None:
        """Test Flatten utility layer."""
        layer = getattr(__import__("illia.nn", fromlist=["Flatten"]), "Flatten")()
        inputs = tf.random.uniform((32, 28, 28, 16))
        output = layer(inputs)
        assert isinstance(output, tf.Tensor)
        assert tuple(output.shape) == (32, 28 * 28 * 16)
        assert output.dtype == inputs.dtype
