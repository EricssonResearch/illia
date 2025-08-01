"""
This module contains the code for Linear Bayesian layer.
"""

# Standard libraries
from typing import Any, Optional

# 3pps
import jax
import jax.numpy as jnp
from flax.nnx.nn import dtypes
from flax.typing import DotGeneralT, PrecisionLike
from jax import lax

# Own modules
from illia.distributions.jax import GaussianDistribution
from illia.nn.jax.base import BayesianModule


class Linear(BayesianModule):
    """
    This class is the bayesian implementation of the Linear class.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
        precision: PrecisionLike = None,
        dot_general: DotGeneralT = lax.dot_general,
    ) -> None:
        """
        This is the constructor of the Linear class.

        Args:
            input_size: Size of the input features.
            output_size: Size of the output features.
            weights_distribution: Prior distribution of the weights.
            bias_distribution: Prior distribution of the bias.
            use_bias: Whether to include a bias term in the layer.
            precision: Precision used in dot product operations.
            dot_general: Function for computing generalized dot
                products.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.backend_params: dict[str, Any] = {
            "use_bias": use_bias,
            "precision": precision,
            "dot_general": dot_general,
        }

        # Set weights prior
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution((input_size, output_size))
        else:
            self.weights_distribution = weights_distribution

        # Set bias prior
        if self.backend_params["use_bias"]:
            if bias_distribution is None:
                self.bias_distribution = GaussianDistribution((output_size,))
            else:
                self.bias_distribution = self.bias_distribution

        return None

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        This method is the forward pass of the model.

        Args:
            inputs: Inputs of the model. Dimensions: [*, input size].

        Returns:
            Output tensor. Dimension: [*, output size].
        """

        # Sample if model not frozen
        if not self.frozen:
            # Sample weights
            self.weights = self.weights_distribution.sample()

            # Sample bias
            if self.backend_params["use_bias"]:
                self.bias = self.bias_distribution.sample()

        # Compute ouputs
        inputs, _, _ = dtypes.promote_dtype(
            (inputs, self.weights, self.bias), dtype=self.dtype  # type: ignore
        )
        outputs = self.dot_general(  # type: ignore
            inputs,
            self.weights,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,  # type: ignore
        )
        if self.backend_params["use_bias"]:
            outputs += jnp.reshape(self.bias, (1,) * (outputs.ndim - 1) + (-1,))

        return outputs

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            KL divergence cost.
            Total number of parameters.
        """

        # Compute log probs
        log_probs: jax.Array = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # Compute the number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params
