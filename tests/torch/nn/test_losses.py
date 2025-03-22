"""
This module contains the code to test losses.
"""

# 3pps
import torch
import pytest

# Own modules
from illia.torch.nn import KLDivergenceLoss, Linear


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

        Returns:
            None.
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

        return None

    @pytest.mark.order(2)
    def test_forward_multiple(
        self, composed_fixture: tuple[torch.nn.Module, torch.Tensor]
    ) -> None:
        """
        This method is the test for the forward pass.

        Args:
            linear_fixture: tuple of instance of Composed model and
                inputs to use.

        Returns:
            None.
        """

        # Get model and inputs
        model: torch.nn.Module
        model, _ = composed_fixture

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

        return None
