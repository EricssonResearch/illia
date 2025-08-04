"""
This module contains the code for Linear Bayesian layer.
"""

# Standard libraries
from typing import Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs
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
        rngs: Rngs = nnx.Rngs(0),
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
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.precision = precision
        self.dot_general = dot_general
        self.rngs = rngs

        # Set weights prior
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                shape=(self.output_size, self.input_size), rngs=self.rngs
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias prior
        if self.use_bias:
            if bias_distribution is None:
                self.bias_distribution = GaussianDistribution(
                    shape=(self.output_size,), rngs=self.rngs
                )
            else:
                self.bias_distribution = bias_distribution
        else:
            self.bias_distribution = None  # type: ignore

        # Sample initial weights
        self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample initial bias only if using bias
        if self.use_bias and self.bias_distribution:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        else:
            self.bias = None  # type: ignore

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:  # type: ignore
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample bias if they are undefined and bias is used
        if self.use_bias and self.bias is None and self.bias_distribution:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))

        # Stop gradient computation (more similar to detach) weights and bias
        self.weights = jax.lax.stop_gradient(self.weights)
        if self.use_bias and self.bias is not None:
            self.bias = jax.lax.stop_gradient(self.bias)

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs for weights
        log_probs: jax.Array = self.weights_distribution.log_prob(
            jnp.asarray(self.weights)
        )

        # Add bias log probs only if using bias
        if self.use_bias and self.bias is not None and self.bias_distribution:
            log_probs += self.bias_distribution.log_prob(jnp.asarray(self.bias))

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params
        if self.use_bias and self.bias_distribution:
            num_params += self.bias_distribution.num_params

        return log_probs, num_params

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
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution:
                self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))

        # Compute outputs
        outputs = inputs @ self.weights.T

        # Add bias only if using bias
        if self.use_bias and self.bias is not None:
            outputs += jnp.reshape(
                jnp.asarray(self.bias), (1,) * (outputs.ndim - 1) + (-1,)
            )

        return outputs
