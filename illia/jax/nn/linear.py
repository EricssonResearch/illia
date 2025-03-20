"""
This module contains the code for Linear Bayesian layer.
"""

# Standard libraries
from typing import Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.typing import (
    Dtype,
    PrecisionLike,
    DotGeneralT,
)

# Own modules
from illia.jax.nn import BayesianModule
from illia.jax.distributions import GaussianDistribution


class Linear(BayesianModule):
    """
    This class is the Linear bayesian layer.

    Attr:
        input_size: input size of the Linear Layer.
        output_size: output size of the Linear layer.
        weights_posterior:

    Returns:
        _description_
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        *,
        use_bias: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        dot_general: DotGeneralT = lax.dot_general,
        rngs: rnglib.Rngs = nnx.Rngs(0),
    ) -> None:
        # call super class constructor
        super().__init__()

        # set attributes
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dot_general = dot_general

        # set weights prior
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution((input_size, output_size))

        else:
            self.weights_distribution = weights_distribution

        # set bias prior
        if bias_distribution is None:
            self.bias_distribution = GaussianDistribution((output_size,))

        else:
            self.bias_distribution = self.bias_distribution

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        This methos is the forward pass of the model.

        Args:
            inputs: inputs of the model. Dimensions: [*, input size].

        Returns:
            output tensor. Dimension: [*, output size].
        """

        # sample if model not frozen
        if not self.frozen:
            # sample
            self.weights: jax.Array = self.weights_distribution.sample()
            self.bias: jax.Array = self.bias_distribution.sample()

        # compute ouputs
        inputs, kernel, bias = dtypes.promote_dtype(
            (inputs, self.weights, self.bias), dtype=self.dtype
        )
        outputs = self.dot_general(
            inputs,
            self.weights,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.bias is not None:
            outputs += jnp.reshape(self.bias, (1,) * (outputs.ndim - 1) + (-1,))

        return outputs

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        This method computes the kl-divergence cost for the layer.

        Returns:
            kl cost.
            number of parameters of the layer.
        """

        # compute log probs
        log_probs: jax.Array = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # compute the number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params
