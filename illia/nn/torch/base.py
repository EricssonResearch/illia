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
        Freezes the layer parameters by stopping gradient computation.
        If the weights or bias are not already sampled, they are sampled
        before freezing. Once frozen, no further sampling occurs.

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
        Computes the KL divergence cost for weights and bias.

        Returns:
            A tuple containing:
                - KL divergence cost.
                - Total number of parameters in the layer.

        Notes:
            Includes bias in the KL computation only if use_bias is
            True.
        """

        return torch.tensor(0.0), 0
