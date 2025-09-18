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
    Bayesian linear (fully connected) layer with optional weight and bias
    priors. Functions like a standard linear layer but treats weights and
    bias as probabilistic variables. Freezing the layer fixes parameters
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
        Initialize a Bayesian linear layer with optional priors for weights
        and bias. Samples initial parameter values from the specified
        distributions.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.
            weights_distribution: Distribution for weights.
            bias_distribution: Distribution for bias.
            use_bias: Whether to include a bias term.
            precision: Precision for dot product computations.
            dot_general: Function for generalized dot products.
            rngs: Random number generators for reproducibility.
            **kwargs: Additional arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
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
        if self.use_bias and self.bias_distribution is not None:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        else:
            self.bias = None

    def freeze(self) -> None:
        """
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample bias if they are undefined and bias is used
        if self.use_bias and self.bias is None and self.bias_distribution is not None:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))

        # Stop gradient computation
        self.weights = jax.lax.stop_gradient(self.weights)
        if self.use_bias:
            self.bias = jax.lax.stop_gradient(self.bias)

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[jax.Array, int]: A tuple containing the KL divergence
                cost and the total number of parameters in the layer.
        """

        # Compute log probs for weights
        log_probs: jax.Array = self.weights_distribution.log_prob(
            jnp.asarray(self.weights)
        )

        # Add bias log probs only if using bias
        if (
            self.use_bias
            and self.bias is not None
            and self.bias_distribution is not None
        ):
            log_probs += self.bias_distribution.log_prob(jnp.asarray(self.bias))

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params
        if self.use_bias and self.bias_distribution is not None:
            num_params += self.bias_distribution.num_params

        return log_probs, num_params

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Perform a forward pass using current weights and bias. Samples new
        parameters if the layer is not frozen.

        Args:
            inputs: Input array with shape [*, input_size].

        Returns:
            Output array with shape [*, output_size].

        Raises:
            ValueError: If the layer is frozen but weights or bias are
                undefined.
        """

        # Sample if model not frozen
        if not self.frozen:
            # Sample weights
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution is not None:
                self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        elif self.weights is None or (self.use_bias and self.bias is None):
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute outputs
        outputs = inputs @ self.weights.T

        # Add bias only if using bias
        if self.use_bias and self.bias is not None:
            outputs += jnp.reshape(
                jnp.asarray(self.bias), (1,) * (outputs.ndim - 1) + (-1,)
            )

        return outputs
