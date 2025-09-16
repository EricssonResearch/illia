"""
This module contains the tests for the bayesian LSTM.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "torch"

# 3pps
import pytest
import torch
from torch.jit import RecursiveScriptModule

# Own modules
from illia.nn import LSTM


class TestLSTM:
    """
    This class tests the bayesian LSTM.
    """

    @pytest.mark.order(1)
    def test_init(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
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
    def test_forward(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
        """
        Test the LSTM forward pass.
        """
        model, inputs = lstm_fixture
        outputs, (hidden_state, cell_state) = model(inputs)

        # Check type of outputs
        assert isinstance(outputs, torch.Tensor), (
            f"Incorrect outputs class, expected {torch.Tensor} "
            f"and got {type(outputs)}"
        )
        assert isinstance(hidden_state, torch.Tensor), (
            f"Incorrect hidden_state class, expected {torch.Tensor} "
            f"and got {type(hidden_state)}"
        )
        assert isinstance(cell_state, torch.Tensor), (
            f"Incorrect cell_state class, expected {torch.Tensor} "
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
    def test_backward(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
        """
        Test the LSTM backward pass.
        """
        model, inputs = lstm_fixture

        # First backward pass
        outputs, _ = model(inputs)
        loss = outputs.sum()
        loss.backward()

        # Check that gradients were computed for embedding parameters
        for name, parameter in model.embedding.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, (
                    "Incorrect backward computation, gradient of embedding."
                    f"{name} shouldn't be None"
                )

        # Clear gradients before second pass
        model.zero_grad()

        # Second backward pass
        outputs, _ = model(inputs)
        loss = outputs.sum()
        loss.backward()

        # Check gradients again
        for name, parameter in model.embedding.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, (
                    "Incorrect backward computation, gradient of embedding."
                    f"{name} shouldn't be None"
                )

    @pytest.mark.order(4)
    def test_freeze(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
        """
        Test the freeze/unfreeze behavior of LSTM.
        """
        model, inputs = lstm_fixture

        # Compute outputs (unfrozen by default)
        outputs_first, _ = model(inputs)
        outputs_second, _ = model(inputs)

        # Check if both outputs are different (stochastic behavior)
        assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect outputs, different forwards are equal when at the "
            "initialization the layer should be unfrozen"
        )

        # Freeze layer
        model.freeze()

        # Compute outputs
        outputs_first, _ = model(inputs)
        outputs_second, _ = model(inputs)

        # Check if both outputs are equal (deterministic behavior)
        assert torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect freezing, when layer is frozen outputs are not the same in "
            "different forward passes"
        )

        # Unfreeze layer
        model.unfreeze()

        # Compute outputs
        outputs_first, _ = model(inputs)
        outputs_second, _ = model(inputs)

        # Check if both outputs are different (stochastic behavior)
        assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect unfreezing, when layer is unfrozen outputs are the same in "
            "different forward passes"
        )

    @pytest.mark.order(5)
    def test_kl_cost(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
        """
        This method is the test for the kl_cost method of LSTM layer.
        """
        model, _ = lstm_fixture

        # Compute KL cost
        kl_cost, num_params = model.kl_cost()

        # Check type of outputs
        assert isinstance(
            kl_cost, torch.Tensor
        ), f"Incorrect KL cost type, expected {torch.Tensor} and got {type(kl_cost)}"
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

    @pytest.mark.order(6)
    def test_jit(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
        """
        This method tests the scripting of the layer.
        """
        model, inputs = lstm_fixture

        # Script the model
        model_scripted: RecursiveScriptModule = torch.jit.script(model)

        # Compute outputs (should be different due to stochastic sampling)
        outputs_first, _ = model_scripted(inputs)
        outputs_second, _ = model_scripted(inputs)

        # Check if both outputs are different (unfrozen by default)
        assert not torch.allclose(
            outputs_first, outputs_second, 1e-8
        ), "Incorrect default freeze with torchscript."

        # Freeze layer
        model_scripted.freeze()

        # Compute outputs (should be the same when frozen)
        outputs_first, _ = model_scripted(inputs)
        outputs_second, _ = model_scripted(inputs)

        # Check if both outputs are equal
        assert torch.allclose(
            outputs_first, outputs_second, 1e-8
        ), "Incorrect freezing with torchscript."

        # Unfreeze layer
        model_scripted.unfreeze()

        # Compute outputs (should be different again)
        outputs_first, _ = model_scripted(inputs)
        outputs_second, _ = model_scripted(inputs)

        # Check if both outputs are different
        assert not torch.allclose(
            outputs_first, outputs_second, 1e-8
        ), "Incorrect unfreezing with torchscript."

        # Test KL cost computation
        kl_cost, num_params = model_scripted.kl_cost()
        assert isinstance(kl_cost, torch.Tensor), "KL cost should be a tensor"
        assert isinstance(num_params, int), "Number of parameters should be an integer"
