"""
This module contains the tests for the bayesian Conv2D.
"""

import os

# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "torch"


import pytest
import torch
from torch.jit import RecursiveScriptModule

from illia.nn import Conv2D


class TestConv2d:
    """
    This class tests the bayesian Conv2D.
    """

    @pytest.mark.order(1)
    def test_init(self, conv2d_fixture: tuple[Conv2D, torch.Tensor]) -> None:
        """
        This method is the test for the Conv2D constructor.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        model: Conv2D
        model, _ = conv2d_fixture

        # Check parameters length
        len_parameters: int = len(list(model.parameters()))
        assert (
            len_parameters == 4
        ), f"Incorrect parameters length, expected 4 and got {len_parameters}"

    @pytest.mark.order(2)
    def test_forward(self, conv2d_fixture: tuple[Conv2D, torch.Tensor]) -> None:
        """
        This method is the test for the Conv2D forward pass.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        inputs: torch.Tensor
        model, inputs = conv2d_fixture

        # Check parameters length
        outputs: torch.Tensor = model(inputs)

        # Check type of outputs
        assert isinstance(
            outputs, torch.Tensor
        ), f"Incorrect outputs class, expected {torch.Tensor} and got {type(outputs)}"

        # Check outputs shape
        assert outputs.shape[:2] == (inputs.shape[0], model.weights.shape[0]), (
            f"Incorrect outputs shape, expected "
            f"{(inputs.shape[0], model.weights.shape[0])} and got {outputs.shape}"
        )

    @pytest.mark.order(3)
    def test_backward(self, conv2d_fixture: tuple[Conv2D, torch.Tensor]) -> None:
        """
        This method is the test for the Conv2D backward pass.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        inputs: torch.Tensor
        model, inputs = conv2d_fixture

        # check parameters length
        outputs: torch.Tensor = model(inputs)
        outputs.sum().backward()

        # Check type of outputs
        for name, parameter in model.named_parameters():
            # check if parameter is none
            assert parameter.grad is not None, (
                f"Incorrect backward computation, gradient of {name} shouldn't be "
                f"None"
            )

    @pytest.mark.order(4)
    def test_freeze(self, conv2d_fixture: tuple[Conv2D, torch.Tensor]) -> None:
        """
        This method is the test for the freeze and unfreeze layers from
        Conv2D layer.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        inputs: torch.Tensor
        model, inputs = conv2d_fixture

        # Compute outputs
        outputs_first: torch.Tensor = model(inputs)
        outputs_second: torch.Tensor = model(inputs)

        # Check if both outputs are equal
        assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect outputs, different forwards are equal when at the "
            "initialization the layer should be unfrozen"
        )

        # Freeze layer
        model.freeze()

        # Compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        # Check if both outputs are equal
        assert torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect freezing, when layer is frozen outputs are not the same in "
            "different forward passes"
        )

        # Unfreeze layer
        model.unfreeze()

        # Compute outputs
        outputs_first = model(inputs)
        outputs_second = model(inputs)

        # Check if both outputs are equal
        assert not torch.allclose(outputs_first, outputs_second, 1e-8), (
            "Incorrect unfreezing, when layer is unfrozen outputs are the same in "
            "different forward passes"
        )

    @pytest.mark.order(5)
    def test_kl_cost(self, conv2d_fixture: tuple[Conv2D, torch.Tensor]) -> None:
        """
        This method is the test for the kl_cost method of Conv2D layer.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        model, _ = conv2d_fixture

        # Compute outputs
        outputs: tuple[torch.Tensor, int] = model.kl_cost()

        # Check type of output
        assert isinstance(
            outputs, tuple
        ), f"Incorrect output type, expected {tuple} and got {type(outputs)}"

        # Check type of kl cost
        assert isinstance(outputs[0], torch.Tensor), (
            f"Incorrect output type in the first element, expected {torch.Tensor} and "
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

    @pytest.mark.order(6)
    def test_jit(self, conv2d_fixture: tuple[Conv2D, torch.Tensor]) -> None:
        """
        This method tests the scripting of the layer.

        Args:
            conv2d_fixture: tuple of instance of Conv2D and inputs to
                use.
        """

        # Get model and inputs
        model: Conv2D
        inputs: torch.Tensor
        model, inputs = conv2d_fixture

        # Script
        model_scripted: RecursiveScriptModule = torch.jit.script(model)

        # Compute outputs
        outputs_first: torch.Tensor = model_scripted(inputs)
        outputs_second: torch.Tensor = model_scripted(inputs)

        # Check if both outputs are equal
        assert not torch.allclose(
            outputs_first, outputs_second, 1e-8
        ), "Incorrect default freeze with torchscript."

        # Freeze layer
        model_scripted.freeze()

        # Compute outputs
        outputs_first = model_scripted(inputs)
        outputs_second = model_scripted(inputs)

        # Check if both outputs are equal
        assert torch.allclose(
            outputs_first, outputs_second, 1e-8
        ), "Incorrect freezing with torchscript."

        # Unfreeze layer
        model_scripted.unfreeze()

        # Compute outputs
        outputs_first = model_scripted(inputs)
        outputs_second = model_scripted(inputs)

        # Check if both outputs are equal
        assert not torch.allclose(
            outputs_first, outputs_second, 1e-8
        ), "Incorrect unfreezing with torchscript."

        # Compute kl cost
        kl_cost, num_params = model_scripted.kl_cost()
