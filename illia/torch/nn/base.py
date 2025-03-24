"""
This module contains the code for the BayesianModule.
"""

# Standard libraries
from abc import ABC

# 3pps
import torch


class BayesianModule(ABC, torch.nn.Module):
    """
    This class serves as the base class for Bayesian modules.
    Any module designed to function as a Bayesian layer should inherit
    from this class.
    """

    def __init__(self):
        """
        Initializes the BayesianModule, setting the frozen state to
        False.
        """

        # Call super class constructor
        super().__init__()

        # Set freeze false by default
        self.frozen: bool = False

        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

        return None

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

        return None

    @torch.jit.export
    def unfreeze(self) -> None:
        """
        Unfreezes the current module and all submodules that are
        instances of BayesianModule. Sets the frozen state to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

        return None

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Abstract method to compute the KL divergence cost.
        Must be implemented by subclasses.

        Returns:
            A tuple containing the KL divergence cost and its
            associated integer value.
        """

        return torch.tensor(0.0), 0
