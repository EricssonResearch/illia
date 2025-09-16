"""
This module defines an abstract base class for Bayesian layers using
PyTorch. It facilitates identifying, freezing, and computing KL
costs for Bayesian-aware modules.
"""

# Standard libraries
from abc import ABC

# 3pps
import torch


class BayesianModule(torch.nn.Module, ABC):
    """
    Abstract base for Bayesian-aware modules in Flax's nnx framework.
    Any Bayesian layer should inherit from this class.
    """

    def __init__(self) -> None:
        """
        Initializes the module with default Bayesian-specific flags.

        Returns:
            None.
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
        Freezes the current module by setting its `frozen` flag to True.
        This flag can be used in derived classes to disable updates.

        Returns:
            None.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    @torch.jit.export
    def unfreeze(self) -> None:
        """
        Unfreezes the current module by setting its `frozen` flag to False.

        Returns:
            None.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler divergence between
        posterior and prior distributions for the module's
        learnable parameters.

        Returns:
            A tuple containing:
                - kl_cost: The Kullback-Leibler divergence.
                - num_params: The number of contributing parameters.
        """

        return torch.tensor(0.0), 0
