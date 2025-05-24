"""
This module contains the code for the Losses.
"""

# Standard libraries
from typing import Literal

# 3pps
import torch

# Own modules
from illia.nn.torch.base import BayesianModule


class KLDivergenceLoss(torch.nn.Module):
    """
    Computes the KL divergence loss for Bayesian modules within a model.
    """

    def __init__(
        self, reduction: Literal["mean"] = "mean", weight: float = 1.0
    ) -> None:
        """
        Initializes the KL Divergence Loss with specified reduction
        method and weight.

        Args:
            reduction: Method to reduce the loss, currently only "mean"
                is supported.
            weight: Scaling factor for the KL divergence loss.
        """

        # call super class constructor
        super().__init__()

        # Set parameters
        self.reduction = reduction
        self.weight = weight

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Computes the KL divergence loss across all Bayesian modules in
        the model.

        Args:
            model: PyTorch model containing Bayesian modules.

        Returns:
            KL divergence cost scaled by the specified weight.
        """

        # Get device and dtype
        parameter: torch.nn.Parameter = next(model.parameters())
        device: torch.device = parameter.device
        dtype = parameter.dtype

        # Init kl cost and params
        kl_global_cost: torch.Tensor = torch.tensor(0, device=device, dtype=dtype)
        num_params_global: int = 0

        # Iter over modules
        for module in model.modules():
            if isinstance(module, BayesianModule):
                kl_cost, num_params = module.kl_cost()
                kl_global_cost += kl_cost
                num_params_global += num_params

        # Average by the number of parameters
        kl_global_cost /= num_params
        kl_global_cost *= self.weight

        return kl_global_cost
