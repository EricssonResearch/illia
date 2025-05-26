"""
This module contains the code to test losses.
"""

# Standard libraries
import os

# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "torch"

# 3pps
import pytest
import torch

# Own modules
from illia.nn import BayesianModule, Linear
from illia.losses import KLDivergenceLoss


class TestKLDivergenceLoss:
    """
    This class implements the tests for KLDivergenceLoss.
    """

    @pytest.mark.order(1)
    def test_forward_single(self, linear_fixture: tuple[Linear, torch.Tensor]) -> None:
        """
        This method is the test for the forward pass.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        model: Linear
        model, _ = linear_fixture

        # Define loss and compute value
        loss: torch.nn.Module = KLDivergenceLoss()
        loss_value: torch.Tensor = loss(model)

        # Check type
        assert isinstance(loss_value, torch.Tensor), (
            f"Incorrect type of loss value, expected {torch.Tensor} and got "
            f"{type(loss_value)}"
        )

        # Check shape
        assert (
            loss_value.shape == ()
        ), f"Incorrect shape, got {loss_value.shape} and got ()"

    @pytest.mark.order(2)
    def test_backward_single(self, linear_fixture: tuple[Linear, torch.Tensor]) -> None:
        """
        This method is the test for the backward pass.

        Args:
            linear_fixture: tuple of instance of Linear and inputs to
                use.
        """

        # Get model and inputs
        model: Linear
        model, _ = linear_fixture

        # Define loss and compute value
        loss: torch.nn.Module = KLDivergenceLoss()
        loss_value: torch.Tensor = loss(model)
        loss_value.backward()

        # Iter over parameters
        for name, parameter in model.named_parameters():
            # Check if parameter is none
            assert parameter.grad is not None, (
                f"Incorrect backward computation, gradient of {name} shouldn't be "
                f"None"
            )

    @pytest.mark.order(3)
    def test_forward_multiple(
        self, composed_fixture: tuple[BayesianModule, torch.nn.Module, torch.Tensor]
    ) -> None:
        """
        This method is the test for the forward pass.

        Args:
            linear_fixture: tuple of instance of Composed model and
                inputs to use.
        """

        # Get model and inputs
        bayesian_model: BayesianModule
        model: torch.nn.Module
        bayesian_model, model, _ = composed_fixture

        # Define loss and compute value
        loss: torch.nn.Module = KLDivergenceLoss()
        loss_value_bayesian: torch.Tensor = loss(bayesian_model)
        loss_value: torch.Tensor = loss(model)

        # Check type
        assert isinstance(loss_value, torch.Tensor), (
            f"Incorrect type of loss value, expected {torch.Tensor} and got "
            f"{type(loss_value)}"
        )

        # Check shape
        assert (
            loss_value.shape == ()
        ), f"Incorrect shape, got {loss_value.shape} and got ()"

        # Check value
        assert torch.allclose(
            loss_value, loss_value_bayesian
        ), "Not equal value with BayesianModule and torch.nn.Module."
