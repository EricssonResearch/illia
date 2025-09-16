"""
This module implements  the Evidence Lower Bound
(ELBO) loss for Bayesian neural networks in PyTorch.
"""

# Standard libraries
from typing import Any

# 3pps
import torch

# Own modules
from illia.losses.torch.kl import KLDivergenceLoss


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
        
        Returns:
            None.
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
