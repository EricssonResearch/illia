"""
This module contains the code for the Losses.
"""

# Standard libraries
from typing import Literal

# 3pps
import torch

# Own modules
from illia.losses.torch import KLDivergenceLoss


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
            outputs: Predicted values from the model.
            targets: True target values.
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
