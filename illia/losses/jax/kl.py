# Standard libraries
from typing import Any, Literal

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx

# Own modules
from illia.nn.jax.base import BayesianModule


class KLDivergenceLoss(nnx.Module):
    """
    Compute Kullback-Leibler divergence across Bayesian modules.
    This loss sums the KL divergence from all Bayesian layers in
    the model. It can be reduced by averaging and scaled by a
    weight factor.

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
        Initialize the KL divergence loss.

        Args:
            reduction: Method used to reduce the KL loss.
            weight: Scaling factor for the KL divergence.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None
        """

        super().__init__(**kwargs)

        self.reduction = reduction
        self.weight = weight

    def __call__(self, model: nnx.Module) -> jax.Array:
        """
        Compute KL divergence for all Bayesian modules in a model.

        Args:
            model: Model containing Bayesian submodules.

        Returns:
            jax.Array: Weighted KL divergence loss.

        Notes:
            The KL loss is averaged over the number of parameters
            and scaled by the `weight` attribute.
        """

        # Init kl cost and params
        kl_global_cost: jax.Array = jnp.array(0.0)
        num_params_global: int = 0

        # Iter over modules
        for _, module in model.iter_modules():
            if isinstance(module, BayesianModule):
                kl_cost, num_params = module.kl_cost()
                kl_global_cost += kl_cost
                num_params_global += num_params

        # Average by the number of parameters
        kl_global_cost /= num_params
        kl_global_cost *= self.weight

        return kl_global_cost
