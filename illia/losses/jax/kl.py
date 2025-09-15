"""
This module implements the Kullback-Leibler (KL) divergence
loss for Bayesian neural networks in Jax.
"""

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
    Computes the KL divergence loss across all Bayesian modules.

    Supports optional weighting and currently only "mean" reduction.
    """

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the KL divergence loss computation.

        Args:
            reduction: Reduction method for the loss. Only "mean"
                supported.
            weight: Scaling factor applied to the total KL loss.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def __call__(self, model: nnx.Module) -> jax.Array:
        """
        Computes KL divergence for all Bayesian modules in the model.

        Args:
            model: NNX model containing Bayesian submodules.

        Returns:
            Scaled KL divergence loss as a scalar array.
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
