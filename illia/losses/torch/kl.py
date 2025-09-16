"""
This module implements the Kullback-Leibler (KL) divergence
loss for Bayesian neural networks in PyTorch.
"""

# Standard libraries
from typing import Any, Literal

# 3pps
import torch

# Own modules
from illia.nn.torch.base import BayesianModule


class KLDivergenceLoss(torch.nn.Module):
    """
    Computes the Kullback-Leibler divergence loss across
    all Bayesian modules.
    """

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Kullback-Leibler divergence loss computation.

        Args:
            reduction: Reduction method for the loss.
            weight: Scaling factor applied to the total KL loss.

        Returns:
            None.
        """

        # call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Computes Kullback-Leibler divergence for all Bayesian
        modules in the model.

        Args:
            model: Model containing Bayesian submodules.

        Returns:
            Scaled Kullback-Leibler divergence loss as a scalar array.
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
