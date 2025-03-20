"""
This module contains the code for the Losses.
"""

# Standard libraries
from typing import Literal

# 3pp
import torch

# Own modules
from illia.torch.nn.base import BayesianModule


class KLDivergenceLoss(torch.nn.Module):
    """
    Computes the KL divergence loss for Bayesian modules within a model.
    """

    def __init__(self, reduction: Literal["mean"] = "mean", weight: float = 1.0):
        """
        Initializes the KL Divergence Loss with specified reduction
        method and weight.

        Args:
            reduction: Method to reduce the loss, currently only "mean"
                is supported.
            weight: Scaling factor for the KL divergence loss.
        """

        # Call super class constructor
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

        kl_global_cost: torch.Tensor = torch.tensor(
            0, device=next(model.parameters()).device, dtype=torch.float32
        )
        num_params_global: int = 0
        for module in model.modules():
            if module != model and isinstance(module, BayesianModule):
                kl_cost, num_params = module.kl_cost()
                kl_global_cost += kl_cost
                num_params_global += num_params

        kl_global_cost /= num_params
        kl_global_cost *= self.weight

        return kl_global_cost


class ELBOLoss(torch.nn.Module):
    """
    Computes the Evidence Lower Bound (ELBO) loss, combining a
    reconstruction loss and KL divergence.
    """

    def __init__(
        self,
        loss_function: torch.nn.Module,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
    ) -> None:
        """
        Initializes the ELBO loss with specified reconstruction loss
        function, sample count, and KL weight.

        Args:
            loss_function: Loss function for computing reconstruction
                loss.
            num_samples: Number of samples for Monte Carlo
                approximation.
            kl_weight: Scaling factor for the KL divergence component.
        """

        # Call super class constructor
        super().__init__()

        self.loss_function = loss_function
        self.num_samples = num_samples
        self.kl_weight = kl_weight
        self.kl_loss = KLDivergenceLoss(weight=kl_weight)

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Computes the ELBO loss, averaging over multiple samples.

        Args:
            y_true: True target values.
            y_pred: Predicted values from the model.
            model: PyTorch model containing Bayesian modules.

        Returns:
            Average ELBO loss across samples.
        """

        loss_value = torch.tensor(
            0, device=next(model.parameters()).device, dtype=torch.float32
        )
        for _ in range(self.num_samples):
            loss_value += self.loss_function(outputs, targets) + self.kl_loss(model)

        loss_value /= self.num_samples

        return loss_value
