"""
This module contains the tests for the Embedding layer.
"""

# Standard libraries
import os
import tempfile

# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "tf"


# 3pps
import keras
import pytest
import tensorflow as tf

# Own modules
from illia.nn import Embedding


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
                f"Incorrect backward computation, gradient of "
                f"{model.trainable_variables[i]} shouldn't be None"
            )

    @pytest.mark.order(4)
    def test_freeze(self, embedding_fixture: tuple[Embedding, tf.Tensor]) -> None:
        """
        This method is the test for the freeze and unfreeze layers from
        Embedding layer.

        Args:
            embedding_fixture: tuple of instance of Embedding and inputs to
                use.
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

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.order(6)
    def test_saving_load_model(
        self, embedding_fixture: tuple[Embedding, tf.Tensor]
    ) -> None:
        """
        This method is the test for the test_saving_load_model of Embedding layer.

        Args:
            embedding_fixture: tuple of instance of Embedding and inputs to use.
        """

        # Get model and inputs
        layer: Embedding
        inputs: tf.Tensor
        layer, inputs = embedding_fixture

        # Create a model
        input_model = keras.Input(shape=inputs.shape[1:])
        output_model = layer(input_model)
        model = keras.Model(input_model, output_model)

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define the path for saving the model
            model_path = f"{temp_dir}/embedding_test.keras"

            # Save the model
            model.save(model_path)

            # Load the model without compiling it
            loaded_model = keras.models.load_model(model_path, compile=False)

            # Verify the loaded model is the same as the original model
            # by checking the architecture and weights
            original_weights = model.get_weights()
            loaded_weights = loaded_model.get_weights()

            # Check if all weights are equal
            for o_w, l_w in zip(original_weights, loaded_weights):
                assert tf.reduce_all(
                    tf.equal(o_w, l_w)
                ), "Weights of the loaded model do not match the original model."

            # Optionally, verify the loaded model with a forward pass
            original_output = model(inputs)
            loaded_output = loaded_model(inputs)

            # Check if the shapes and dtypes of the outputs are the same
            assert (
                original_output.shape == loaded_output.shape
            ), "Incorrect shape of the loaded model outputs."
            assert (
                original_output.dtype == loaded_output.dtype
            ), "Incorrect dtype of the loaded model outputs."
