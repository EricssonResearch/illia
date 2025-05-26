"""
This module contains the code to test losses.
"""

# Standard libraries
import os

# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "tf"

# 3pps
import pytest
import keras
import tensorflow as tf

# Own modules
from illia.nn import Linear
from illia.losses import KLDivergenceLoss


class TestKLDivergenceLoss:
    """
    This class implements the tests for KLDivergenceLoss.
    """

    @pytest.mark.order(1)
    def test_forward_single(self, linear_fixture: tuple[Linear, tf.Tensor]) -> None:
        """
        This method is the test for the forward pass.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        linear_layer: Linear
        linear_layer, _ = linear_fixture
        model = keras.Sequential([linear_layer])

        # Define loss and compute value
        loss: keras.losses.Loss = KLDivergenceLoss()
        loss_value: tf.Tensor = loss(model=model)

        # Check type
        assert isinstance(loss_value, tf.Tensor), (
            f"Incorrect type of loss value, expected {tf.Tensor} and got "
            f"{type(loss_value)}"
        )

        # Check shape
        assert (
            loss_value.shape == ()
        ), f"Incorrect shape, got {loss_value.shape} and got ()"

    @pytest.mark.order(2)
    def test_backward_single(self, linear_fixture: tuple[Linear, tf.Tensor]) -> None:
        """
        This method is the test for the backward pass.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        linear_layer: Linear
        linear_layer, _ = linear_fixture
        model = keras.Sequential([linear_layer])

        # Define loss and compute value
        loss: keras.losses.Loss = KLDivergenceLoss()

        with tf.GradientTape() as tape:
            loss_value: tf.Tensor = loss(model=model)
        gradients = tape.gradient(loss_value, model.trainable_variables)

        # Check type of outputs
        for i, gradient in enumerate(gradients):
            # Check if parameter is none
            assert gradient is not None, (
                f"Incorrect backward computation, gradient of "
                f"{model.trainable_variables[i]} shouldn't be None"
            )
