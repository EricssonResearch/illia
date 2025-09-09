# """
# This module contains the tests for the bayesian LSTM.
# """

# # Standard libraries
# import os


# # Change Illia Backend
# os.environ["ILLIA_BACKEND"] = "torch"

# # 3pps
# import pytest
# import torch
# from torch.jit import RecursiveScriptModule

# # Own modules
# from illia.nn import LSTM


# class TestLSTM:
#     """
#     This class tests the bayesian LSTM.
#     """

#     @pytest.mark.order(1)
#     def test_init(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
#         """
#         Test the LSTM constructor.
#         """

#         model, _ = lstm_fixture

#         len_parameters: int = len(list(model.parameters()))
#         assert len_parameters > 0, "LSTM must have at least one trainable parameter"

#     @pytest.mark.order(2)
#     def test_forward(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
#         """
#         Test the LSTM forward pass.
#         """

#         model, inputs = lstm_fixture
#         outputs: torch.Tensor = model(inputs)

#         # Check type of outputs
#         assert isinstance(
#             outputs, torch.Tensor
#         ), f"Incorrect outputs class, expected {torch.Tensor} and got {type(outputs)}"

#         if model.batch_first:
#             # (batch, seq_len, hidden_size)
#             assert (
#                 outputs.shape[0] == inputs.shape[0]
#             ), f"Expected batch size {inputs.shape[0]}, got {outputs.shape[0]}"
#             assert (
#                 outputs.shape[1] == inputs.shape[1]
#             ), f"Expected seq_len {inputs.shape[1]}, got {outputs.shape[1]}"
#         else:
#             # (seq_len, batch, hidden_size)
#             assert (
#                 outputs.shape[0] == inputs.shape[0]
#             ), f"Expected seq_len {inputs.shape[0]}, got {outputs.shape[0]}"
#             assert (
#                 outputs.shape[1] == inputs.shape[1]
#             ), f"Expected batch size {inputs.shape[1]}, got {outputs.shape[1]}"

#         assert (
#             outputs.shape[-1] == model.hidden_size
#         ), f"Expected hidden_size {model.hidden_size}, got {outputs.shape[-1]}"

#     @pytest.mark.order(3)
#     def test_backward(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
#         """
#         Test the LSTM backward pass.
#         """
#         model, inputs = lstm_fixture

#         # First backward pass
#         outputs: torch.Tensor = model(inputs)
#         outputs.sum().backward()

#         # Check that gradients were computed
#         for name, parameter in model.named_parameters():
#             assert (
#                 parameter.grad is not None
#             ), f"Incorrect backward computation, gradient of {name} shouldn't be None"

#         # Clear gradients before second pass
#         model.zero_grad()

#         # Second backward pass
#         outputs = model(inputs)
#         outputs.sum().backward()

#         # Check gradients again
#         for name, parameter in model.named_parameters():
#             assert (
#                 parameter.grad is not None
#             ), f"Incorrect backward computation, gradient of {name} shouldn't be None"

#     @pytest.mark.order(4)
#     def test_freeze(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
#         """
#         Test the freeze/unfreeze behavior of LSTM.
#         """

#         # Get model and inputs
#         model: LSTM
#         inputs: torch.Tensor
#         model, inputs = lstm_fixture

#         # Compute outputs
#         outputs_first: torch.Tensor = model(inputs)
#         outputs_second: torch.Tensor = model(inputs)

#         # Check if both outputs are equal
#         assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
#             "Incorrect outputs, different forwards are equal when at the "
#             "initialization the layer should be unfrozen"
#         )

#         # Freeze layer
#         model.freeze()

#         # Compute outputs
#         outputs_first = model(inputs)
#         outputs_second = model(inputs)

#         # Check if both outputs are equal
#         assert torch.allclose(outputs_first, outputs_second, 1e-8), (
#             "Incorrect freezing, when layer is frozen outputs are not the same in "
#             "different forward passes"
#         )

#         # Unfreeze layer
#         model.unfreeze()

#         # Compute outputs
#         outputs_first = model(inputs)
#         outputs_second = model(inputs)

#         # Check if both outputs are equal
#         assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
#             "Incorrect unfreezing, when layer is unfrozen outputs are the same in "
#             "different forward passes"
#         )

#     @pytest.mark.order(5)
#     def test_kl_cost(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
#         """
#         This method is the test for the kl_cost method of LSTM layer.

#         Args:
#             linear_fixture: tuple of instance of LSTM and inputs to
#                 use.
#         """

#         # Get model and inputs
#         model: LSTM
#         model, _ = lstm_fixture

#         # Compute outputs
#         outputs: tuple[torch.Tensor, int] = model.kl_cost()

#         # Check type of output
#         assert isinstance(
#             outputs, tuple
#         ), f"Incorrect output type, expected {tuple} and got {type(outputs)}"

#         # Check type of kl cost
#         assert isinstance(outputs[0], torch.Tensor), (
#             f"Incorrect output type in the first element, expected {torch.Tensor} and " # type: ignore # noqa: E501
#             f"got {type(outputs[0])}"
#         )

#         # Check type of num params
#         assert isinstance(outputs[1], int), (
#             f"Incorrect output type in the second element, expected {int} and got "
#             f"{type(outputs[1])}"
#         )

#         # Check shape of kl cost
#         assert outputs[0].shape == (), (
#             f"Incorrect shape of outputs first element, expected () and got "
#             f"{outputs[0].shape}"
#         )

#     @pytest.mark.order(6)
#     def test_jit(self, lstm_fixture: tuple[LSTM, torch.Tensor]) -> None:
#         """
#         This method tests the scripting of the layer.

#         Args:
#             linear_fixture: tuple of instance of LSTM and inputs to
#                 use.
#         """

#         # Get model and inputs
#         model: LSTM
#         inputs: torch.Tensor
#         model, inputs = lstm_fixture

#         # Script
#         model_scripted: RecursiveScriptModule = torch.jit.script(model)

#         # Compute outputs
#         outputs_first: torch.Tensor = model_scripted(inputs)
#         outputs_second: torch.Tensor = model_scripted(inputs)

#         # Check if both outputs are equal
#         assert not torch.allclose(
#             outputs_first, outputs_second, 1e-8
#         ), "Incorrect default freeze with torchscript."

#         # Freeze layer
#         model_scripted.freeze()

#         # Compute outputs
#         outputs_first = model_scripted(inputs)
#         outputs_second = model_scripted(inputs)

#         # Check if both outputs are equal
#         assert torch.allclose(
#             outputs_first, outputs_second, 1e-8
#         ), "Incorrect freezing with torchscript."

#         # Unfreeze layer
#         model_scripted.unfreeze()

#         # Compute outputs
#         outputs_first = model_scripted(inputs)
#         outputs_second = model_scripted(inputs)

#         # Check if both outputs are equal
#         assert not torch.allclose(
#             outputs_first, outputs_second, 1e-8
#         ), "Incorrect unfreezing with torchscript."

#         # Compute kl cost
#         kl_cost, num_params = model_scripted.kl_cost()
