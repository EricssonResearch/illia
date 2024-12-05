# Libraries
from typing import Literal

import torch

from . import BayesianModule


class KLDivergenceLoss(torch.nn.Module):

    reduction: Literal["mean"]
    weight: float

    def __init__(self, reduction: Literal["mean"] = "mean", weight: float = 1.0):
        """
        Definition of the KL Divergence Loss function.

        Args:
            reduction: Specifies the reduction to apply to the output.
            weight: Weight for the loss.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Computes the forward pass of the KL Divergence Loss for a given model.

        Args:
            model: The model for which the KL Divergence Loss needs to be computed.

        Returns:
            The computed KL Divergence Loss for the given model.
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

        kl_global_cost /= num_params_global
        kl_global_cost *= self.weight

        return kl_global_cost


class ELBOLoss(torch.nn.Module):

    def __init__(
        self,
        loss_function: torch.nn.Module,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
    ) -> None:
        """
        Initializes the Evidence Lower Bound (ELBO) loss function.

        Args:
            loss_function: The loss function to be used for computing the reconstruction loss.
            num_samples: The number of samples to draw for estimating the ELBO.
            kl_weight: The weight applied to the KL divergence.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.loss_function = loss_function
        self.num_samples = num_samples
        self.kl_weight = kl_weight
        self.kl_loss = KLDivergenceLoss(weight=kl_weight)

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Computes the forward pass of the ELBO (Evidence Lower Bound) loss.

        Parameters:
            y_true: The true target values.
            y_pred: The predicted values.
            model: The model to compute the ELBO loss for.

        Returns:
            The computed ELBO loss.
        """

        loss_value = torch.tensor(
            0, device=next(model.parameters()).device, dtype=torch.float32
        )

        for _ in range(self.num_samples):
            loss_value += self.loss_function(y_true, y_pred) + self.kl_loss(model)

        loss_value /= self.num_samples

        return loss_value
