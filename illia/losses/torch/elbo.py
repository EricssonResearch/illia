"""
This module contains the code for the Losses.

Implements the KL divergence loss and the Evidence Lower Bound (ELBO)
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


class ELBOLoss(torch.nn.Module):
    """
    Computes the Evidence Lower Bound (ELBO) loss.

    Combines a reconstruction loss with KL divergence regularization,
    optionally using Monte Carlo sampling.
    """

    def __init__(
        self,
        loss_function: torch.nn.Module,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ELBO loss function.

        Args:
            loss_function: Callable loss used for reconstruction (e.g. MSELoss).
            num_samples: Number of samples for Monte Carlo estimation.
            kl_weight: Scalar weight for the KL divergence term.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.loss_function = loss_function
        self.num_samples = num_samples
        self.kl_weight = kl_weight
        self.kl_loss = KLDivergenceLoss(weight=kl_weight)

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Computes the ELBO loss over a number of samples.

        Args:
            outputs: Model predictions.
            targets: Ground truth values.
            model: A PyTorch model containing BayesianModule layers.

        Returns:
            Average ELBO loss across all samples.
        """

        loss_value = torch.tensor(
            0, device=next(model.parameters()).device, dtype=torch.float32
        )
        for _ in range(self.num_samples):
            loss_value += self.loss_function(outputs, targets) + self.kl_loss(model)

        loss_value /= self.num_samples

        return loss_value
