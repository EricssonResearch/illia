"""
This module contains the tests for the Non-Bayesian layers Wrapper.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "torch"

# 3pps
import pytest
import torch


class TestNonParametricLayers:
    """
    This class tests the non-parametric layers integration.
    """

    @pytest.mark.order(1)
    @pytest.mark.parametrize(
        "layer_name,kwargs,input_shape,expected_shape",
        [
            ("MaxPool2d", {"kernel_size": 2}, (32, 16, 28, 28), (32, 16, 14, 14)),
            ("AvgPool2d", {"kernel_size": 2}, (32, 16, 28, 28), (32, 16, 14, 14)),
            ("MaxPool1d", {"kernel_size": 2}, (32, 16, 64), (32, 16, 32)),
            ("AvgPool1d", {"kernel_size": 2}, (32, 16, 64), (32, 16, 32)),
            (
                "AdaptiveAvgPool2d",
                {"output_size": (1, 1)},
                (32, 16, 28, 28),
                (32, 16, 1, 1),
            ),
            (
                "AdaptiveMaxPool2d",
                {"output_size": (1, 1)},
                (32, 16, 28, 28),
                (32, 16, 1, 1),
            ),
        ],
    )
    def test_pooling(
        self, layer_name: str, kwargs: dict, input_shape: tuple, expected_shape: tuple
    ) -> None:
        """Test pooling layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)(
            **kwargs
        )
        inputs = torch.rand(input_shape)
        output = layer(inputs)
        assert isinstance(output, torch.Tensor)
        assert tuple(output.shape) == expected_shape
        assert output.dtype == inputs.dtype

    @pytest.mark.order(2)
    @pytest.mark.parametrize(
        "layer_name", ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "GELU"]
    )
    def test_activation(self, layer_name: str) -> None:
        """Test activation layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)()
        inputs = torch.rand((32, 16, 28, 28))
        output = layer(inputs)
        assert isinstance(output, torch.Tensor)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(3)
    @pytest.mark.parametrize(
        "layer_name,kwargs,input_shape",
        [
            ("BatchNorm1d", {"num_features": 16}, (32, 16, 64)),
            ("BatchNorm2d", {"num_features": 16}, (32, 16, 28, 28)),
            ("LayerNorm", {"normalized_shape": [16, 28, 28]}, (32, 16, 28, 28)),
        ],
    )
    def test_normalization(
        self, layer_name: str, kwargs: dict, input_shape: tuple
    ) -> None:
        """Test normalization layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)(
            **kwargs
        )
        layer.train()
        inputs = torch.randn(input_shape) * 2.0 + 5.0
        output = layer(inputs)
        assert isinstance(output, torch.Tensor)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype
        # assert torch.abs(torch.mean(output)) < 0.1
        # assert torch.abs(torch.std(output) - 1.0) < 0.1

    @pytest.mark.order(4)
    @pytest.mark.parametrize(
        "layer_name,rate,input_shape",
        [
            ("Dropout", 0.5, (32, 16, 28, 28)),
            ("Dropout2d", 0.5, (32, 16, 28, 28)),
        ],
    )
    def test_regularization(
        self, layer_name: str, rate: float, input_shape: tuple
    ) -> None:
        """Test regularization layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)(rate)
        layer.train()
        inputs = torch.rand(input_shape)
        output = layer(inputs)
        assert isinstance(output, torch.Tensor)
        assert tuple(output.shape) == tuple(inputs.shape)
        assert output.dtype == inputs.dtype

    @pytest.mark.order(5)
    @pytest.mark.parametrize(
        "layer_name,input_shape,expected_shape",
        [
            ("Flatten", (32, 16, 28, 28), (32, 16 * 28 * 28)),
            ("Identity", (32, 16, 28, 28), (32, 16, 28, 28)),
        ],
    )
    def test_utility(
        self, layer_name: str, input_shape: tuple, expected_shape: tuple
    ) -> None:
        """Test utility layers."""
        layer = getattr(__import__("illia.nn", fromlist=[layer_name]), layer_name)()
        inputs = torch.rand(input_shape)
        output = layer(inputs)
        assert isinstance(output, torch.Tensor)
        assert tuple(output.shape) == expected_shape
        assert output.dtype == inputs.dtype
