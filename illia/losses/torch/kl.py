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
    Computes the KL divergence loss across all Bayesian modules in a model.

    Supports reduction and scaling by a configurable weight.
    """

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the KL divergence loss.

        Args:
            reduction: Reduction method for the loss.
                Only "mean" is currently supported.
            weight: Scalar to scale the KL divergence term.
        
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
        Computes KL divergence loss across Bayesian modules in the model.

        Args:
            model: A PyTorch model containing BayesianModule instances.

        Returns:
            KL divergence loss scaled by the given weight.
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
