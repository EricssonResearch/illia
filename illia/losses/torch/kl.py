# Standard libraries
from typing import Any, Literal

# 3pps
import torch

# Own modules
from illia.nn.torch.base import BayesianModule


class KLDivergenceLoss(torch.nn.Module):
    """
    Computes Kullback-Leibler divergence across Bayesian modules.
    This loss sums the KL divergence from all Bayesian layers in the
    model. It can be reduced by averaging and scaled by a weight factor.

    Notes:
        Assumes the model contains submodules derived from
        `BayesianModule`.
    """

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the KL divergence loss computation.

        Args:
            reduction: Method for reducing the KL loss.
            weight: Scaling factor applied to the total KL loss.
            **kwargs: Additional arguments passed to the base class.

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
        Compute KL divergence for all Bayesian modules in a model.

        Args:
            model: Model containing Bayesian submodules.

        Returns:
            Scalar array representing the weighted KL divergence loss.

        Notes:
            The loss is averaged over the number of parameters and
            scaled by the `weight` attribute.
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
