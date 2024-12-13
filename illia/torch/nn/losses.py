# standard libraries
from typing import Literal

# 3pp
import torch

# own modules
from illia.torch.nn.base import BayesianModule


class KLDivergenceLoss(torch.nn.Module):
    def __init__(
        self, reduction: Literal["mean"] = "mean", weight: float = 1.0
    ) -> None:
        """
        This method is the KLDivergenceLoss class constructor.

        Args:
            reduction: reduction function to use in the computation.
                Defaults to "mean".
            weight: weight to ponderate it. Defaults to 1.0.
        """

        # call super class constructor
        super().__init__()

        # set parameters
        self.reduction = reduction
        self.weight = weight

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        """
        This method computes the forward for KLDivergenceLoss

        Args:
            model: torch model.

        Returns:
            kl divergence cost. Dimensions: [].
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
    def __init__(
        self,
        loss_function: torch.nn.Module,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
    ) -> None:
        super().__init__()

        self.loss_function = loss_function
        self.num_samples = num_samples
        self.kl_weight = kl_weight
        self.kl_loss = KLDivergenceLoss(weight=kl_weight)

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        loss_value = torch.tensor(
            0, device=next(model.parameters()).device, dtype=torch.float32
        )
        for _ in range(self.num_samples):
            loss_value += self.loss_function(outputs, targets) + self.kl_loss(model)

        loss_value /= self.num_samples

        return loss_value
