"""
This module contains the tests for the Embedding layer.
"""

# 3pps
import tensorflow as tf
import pytest

# own modules
from illia.tf.nn import Embedding


class TestEmbedding:
    """
    This class tests the bayesian Embedding.
    """

    @pytest.mark.order(1)
    def test_init(self, embedding_fixture: tuple[Embedding, tf.Tensor]) -> None:
        """
        This method is the test for the Embedding constructor.

        Args:
            embedding_fixture: tuple of instance of Embedding and inputs to
                use.

        Returns:
            None.
        """

        model: Embedding
        model, _ = embedding_fixture

        # Check parameters length
        len_parameters: int = len(model.trainable_variables)
        assert (
            len_parameters == 2
        ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    @pytest.mark.order(2)
    def test_forward(self, embedding_fixture: tuple[Embedding, tf.Tensor]) -> None:
        """
        This method is the test for the Embedding forward pass.

        Args:
            embedding_fixture: tuple of instance of Embedding and inputs to
                use.

        Returns:
            None.
        """

        # Get model and inputs
        model: Embedding
        inputs: tf.Tensor
        model, inputs = embedding_fixture

        # Check parameters length
        outputs: tf.Tensor = model(inputs)

        # Check type of outputs
        assert isinstance(
            outputs, tf.Tensor
        ), f"Incorrect outputs class, expected {tf.Tensor} and got {type(outputs)}"

        # Check outputs shape
        assert outputs.shape == (inputs.shape[0], model.w.shape[1]), (
            f"Incorrect outputs shape, expected "
            f"{(inputs.shape[0], model.w.shape[1])} and got {outputs.shape}"
        )

    @pytest.mark.order(3)
    def test_backward(self, embedding_fixture: tuple[Embedding, tf.Tensor]) -> None:
        """
        This method is the test for the Embedding backward pass.

        Args:
            embedding_fixture: tuple of instance of Embedding and inputs to
                use.

        Returns:
            None.
        """

        # Get model and inputs
        model: Embedding
        inputs: tf.Tensor
        model, inputs = embedding_fixture

        # Check parameters length
        with tf.GradientTape() as tape:
            outputs: tf.Tensor = model(inputs)
        gradients = tape.gradient(outputs, model.trainable_variables)

        # Check type of outputs
        for i, gradient in enumerate(gradients):
            # Check if parameter is none
            assert gradient is not None, (
                f"Incorrect backward computation, gradient of {model.trainable_variables[i]} shouldn't be "
                f"None"
            )

    @pytest.mark.order(4)
    def test_freeze(self, embedding_fixture: tuple[Embedding, tf.Tensor]) -> None:
        """
        This method is the test for the freeze and unfreeze layers from
        Embedding layer.

        Args:
            embedding_fixture: tuple of instance of Embedding and inputs to
                use.

        Returns:
            None.
        """

        # Get model and inputs
        model: Embedding
        inputs: tf.Tensor
        model, inputs = embedding_fixture

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
    def test_kl_cost(self, embedding_fixture: tuple[Embedding, tf.Tensor]) -> None:
        """
        This method is the test for the kl_cost method of Embedding layer.

        Args:
            embedding_fixture: tuple of instance of Embedding and inputs to
                use.

        Returns:
            None.
        """

        # Get model and inputs
        model: Embedding
        model, _ = embedding_fixture

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
