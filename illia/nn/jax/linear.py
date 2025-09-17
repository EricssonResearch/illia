# Standard libraries
from typing import Any, Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs
from flax.typing import DotGeneralT, PrecisionLike
from jax import lax

# Own modules
from illia.distributions.jax.gaussian import GaussianDistribution
from illia.nn.jax.base import BayesianModule


class Linear(BayesianModule):
    """
    Bayesian linear layer with optional bias and weight priors.
    Functions like a standard fully connected layer but treats weights and
    bias as probabilistic variables. Freezing the layer fixes the parameters
    and stops gradient computation.
    """

    bias_distribution: Optional[GaussianDistribution] = None
    bias: Optional[nnx.Param] = None

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
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Linear layer.

        Args:
            input_size: Size of the input features.
            output_size: Size of the output features.
            weights_distribution: Prior distribution of the weights.
            bias_distribution: Prior distribution of the bias.
            use_bias: Whether to include a bias term in the layer.
            precision: Precision used in dot product operations.
            dot_general: Function for computing generalized dot
                products.

        Returns:
            None.

        Notes:
            If distributions are not provided, Gaussian distributions are
            used by default.
        """

        # Call super class constructor
        super().__init__(**kwargs)

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
            self.bias_distribution = None

        # Sample initial weights
        self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample initial bias only if using bias
        if self.use_bias and self.bias_distribution:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        else:
            self.bias = None

    def freeze(self) -> None:
        """
        Freezes the layer parameters by stopping gradient computation.
        If the weights or bias are not already sampled, they are sampled
        before freezing. Once frozen, no further sampling occurs.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample bias if they are undefined and bias is used
        if self.use_bias and self.bias is None and self.bias_distribution:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))

        # Stop gradient computation
        self.weights = jax.lax.stop_gradient(self.weights)
        if self.use_bias and self.bias:
            self.bias = jax.lax.stop_gradient(self.bias)

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the KL divergence cost for weights and bias.

        Returns:
            A tuple containing:
                - KL divergence cost.
                - Total number of parameters in the layer.

        Notes:
            Includes bias in the KL computation only if use_bias is
            True.
        """

        # Compute log probs for weights
        log_probs: jax.Array = self.weights_distribution.log_prob(
            jnp.asarray(self.weights)
        )

        # Add bias log probs only if using bias
        if self.use_bias and self.bias and self.bias_distribution:
            log_probs += self.bias_distribution.log_prob(jnp.asarray(self.bias))

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params
        if self.use_bias and self.bias_distribution:
            num_params += self.bias_distribution.num_params

        return log_probs, num_params

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Performs a forward pass using current weights and bias.

        Args:
            inputs: Input array with shape [*, input_size].

        Returns:
            Output array with shape [*, output_size].

        Notes:
            If the layer is not frozen, new weights and bias are sampled
            before computation.
        """

        # Sample if model not frozen
        if not self.frozen:
            # Sample weights
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution:
                self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        elif self.weights is None or (self.use_bias and self.bias is None):
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute outputs
        outputs = inputs @ self.weights.T

        # Add bias only if using bias
        if self.use_bias and self.bias:
            outputs += jnp.reshape(
                jnp.asarray(self.bias), (1,) * (outputs.ndim - 1) + (-1,)
            )

        return outputs
