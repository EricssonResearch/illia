"""
This module contains the tests for the bayesian Conv2D.
"""

# 3pps
import keras
import tensorflow as tf
import pytest

# Own modules
from illia.tf.nn import Conv2D


class TestConv2d:
    """
    This class tests the bayesian Conv2D.
    """

    @pytest.mark.order(1)
    def test_init(self, conv2d_fixture: tuple[Conv2D, tf.Tensor, str]) -> None:
        """
        This method is the test for the Conv2D constructor.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        model: Conv2D
        model, _, _ = conv2d_fixture

        # Check parameters length
        len_parameters: int = len(model.trainable_variables)
        assert (
            len_parameters == 4
        ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    @pytest.mark.order(2)
    def test_forward(self, conv2d_fixture: tuple[Conv2D, tf.Tensor, str]) -> None:
        """
        This method is the test for the Conv2D forward pass.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        inputs: tf.Tensor
        model, inputs, data_format = conv2d_fixture

        # Check parameters length
        outputs: tf.Tensor = model(inputs)

        # Check type of outputs
        assert isinstance(
            outputs, tf.Tensor
        ), f"Incorrect outputs class, expected {tf.Tensor} and got {type(outputs)}"

        # Check outputs shape
        if data_format == "NHWC":
            assert (outputs.shape[0], outputs.shape[-1]) == (
                inputs.shape[0],
                model.w.shape[-1],
            ), (
                f"Incorrect outputs shape, expected "
                f"{(inputs.shape[0], model.w.shape[-1])} and got "
                f"{(outputs.shape[0], outputs.shape[-1])}"
            )
        elif data_format == "NCHW":
            assert (outputs.shape[0], outputs.shape[1]) == (
                inputs.shape[0],
                model.w.shape[-1],
            ), (
                f"Incorrect outputs shape, expected "
                f"{(inputs.shape[0], model.w.shape[-1])} and got "
                f"{(outputs.shape[0], outputs.shape[1])}"
            )
        else:
            raise ValueError(f"Invalid data format: {data_format}")

    @pytest.mark.order(3)
    def test_backward(self, conv2d_fixture: tuple[Conv2D, tf.Tensor, str]) -> None:
        """
        This method is the test for the Conv2D backward pass.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        inputs: tf.Tensor
        model, inputs, _ = conv2d_fixture

        # Skip gradient test if running on CPU
        if len(tf.config.list_physical_devices("GPU")) == 0:
            pytest.skip(
                "Skipping backward test because grouped convolution "
                "gradients are not supported on CPU."
            )

        # Check parameters length
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs: tf.Tensor = model(inputs)
        gradients = tape.gradient(outputs, model.trainable_variables)

        # Check type of outputs
        for i, gradient in enumerate(gradients):
            # Check if parameter is none
            assert gradient is not None, (
                "Incorrect backward computation, gradient of "
                f"{model.trainable_variables[i]} shouldn't be None"
            )

    @pytest.mark.order(4)
    def test_freeze(self, conv2d_fixture: tuple[Conv2D, tf.Tensor, str]) -> None:
        """
        This method is the test for the freeze and unfreeze layers from
        Conv2D layer.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        inputs: tf.Tensor
        model, inputs, _ = conv2d_fixture

        # Compute outputs
        outputs_first: tf.Tensor = model(inputs)
        outputs_second: tf.Tensor = model(inputs)

        # Check if both outputs are equal
        assert not tf.experimental.numpy.allclose(
            outputs_first, outputs_second, 1e-8
        ), (
            "Incorrect outputs, different forwards are equal when at the "
            "initialization the layer should be unfrozen"
        )

        # Freeze layer
        model.freeze()

        # Compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        # Check if both outputs are equal
        assert tf.experimental.numpy.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect freezing, when layer is frozen outputs are not the same in "
            "different forward passes"
        )

        # Unfreeze layer
        model.unfreeze()

        # Compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        # Check if both outputs are equal
        assert not tf.experimental.numpy.allclose(
            outputs_first, outputs_second, 1e-8
        ), (
            "Incorrect unfreezing, when layer is unfrozen outputs are the same in "
            "different forward passes"
        )

    @pytest.mark.order(5)
    def test_kl_cost(self, conv2d_fixture: tuple[Conv2D, tf.Tensor, str]) -> None:
        """
        This method is the test for the kl_cost method of Conv2D layer.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        model, _, _ = conv2d_fixture

        # Compute outputs
        outputs: tuple[tf.Tensor, int] = model.kl_cost()

        # Check type of output
        assert isinstance(
            outputs, tuple
        ), f"Incorrect output type, expected {tuple} and got {type(outputs)}"

        # Check type of kl cost
        assert isinstance(outputs[0], tf.Tensor), (
            f"Incorrect output type in the first element, expected {tf.Tensor} and "
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
