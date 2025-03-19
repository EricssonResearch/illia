"""
This module contains the code for the BayesianModule.
"""

# Standard libraries
from abc import ABC

# 3pps
import torch


class BayesianModule(ABC, torch.nn.Module):
    """
    This class implements a BayesianModule. This is still an abstract
    class since it does not implement the forward method.

    Attributes:
        frozen: Indicator if this layer is frozen or not.
        is_bayesian: Indicator if this layer is bayesian or not.
    """

    def __init__(self) -> None:
        """
        This method is the constructor for BayesianModule.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set state
        self.frozen: bool = False

        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

        return None

    @torch.jit.export
    def freeze(self) -> None:
        """
        This method freezes the layer.

        Returns:
            None.
        """

        # Change state
        self.frozen = True

        return None

    @torch.jit.export
    def unfreeze(self) -> None:
        """
        This method unfreezes the layer.

        Returns:
            None.
        """

        # Change state
        self.frozen = False

        return None

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        This is a default implementation of the kl_cots function,
        which computes.

        Returns:
            Tensor with the kl cost.
            Number of parameters of the layer.
        """

        return torch.tensor([0.0]), 0
