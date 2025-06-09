"""
This module contains the code for the BayesianModule.
"""

from abc import ABC

import torch


class BayesianModule(torch.nn.Module, ABC):
    """
    This class serves as the base class for Bayesian modules.
    Any module designed to function as a Bayesian layer should inherit
    from this class.
    """

    def __init__(self):
        """
        This method is the constructor for BayesianModule.
        """

        # Call super class constructor
        super().__init__()

        # Set freeze false by default
        self.frozen: bool = False

        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    @torch.jit.export
    def unfreeze(self) -> None:
        """
        Unfreezes the current module and all submodules that are
        instances of BayesianModule. Sets the frozen state to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        return torch.tensor(0.0), 0
