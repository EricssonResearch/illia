# Standard libraries
from typing import Any, Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs

# Own modules
from illia.distributions.jax.gaussian import GaussianDistribution
from illia.nn.jax.base import BayesianModule


class Embedding(BayesianModule):
    """
    Bayesian embedding layer with optional padding and max-norm constraints.
    Each embedding vector is sampled from a specified weight distribution.
    If the layer is frozen, embeddings are fixed and gradients are stopped.
    """

    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        weights_distribution: Optional[GaussianDistribution] = None,
        rngs: Rngs = nnx.Rngs(0),
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Bayesian embedding layer with optional constraints.
        Sets up the embedding weight distribution and samples initial values.

        Args:
            num_embeddings: Size of the embedding dictionary.
            embeddings_dim: Dimension of each embedding vector.
            padding_idx: Index whose embeddings are ignored in gradient.
            max_norm: Maximum norm for each embedding vector.
            norm_type: p value for the p-norm in max_norm option.
            scale_grad_by_freq: Scale gradients by inverse word frequency.
            weights_distribution: Distribution to initialize embeddings.
            rngs: Random number generators for reproducibility.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.rngs = rngs

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                (self.num_embeddings, self.embeddings_dim)
            )
        else:
            self.weights_distribution = weights_distribution

        # Sample initial weights
        self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

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

        # Stop gradient computation
        self.weights = jax.lax.stop_gradient(self.weights)

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

        # get number of parameters
        num_params: int = self.weights_distribution.num_params

        return log_probs, num_params

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Perform a forward pass using current embedding weights.

        Args:
            inputs: Array of indices into the embedding matrix.

        Returns:
            Array of shape [*, embeddings_dim] containing the embedding
            vectors corresponding to the input indices.

        Raises:
            ValueError: If the layer is frozen but weights are
                undefined.

        Notes:
            Embeddings at padding_idx are zeroed out, and vectors exceeding
            max_norm are renormalized if specified.
        """

        # Sample if model not frozen
        if not self.frozen:
            # Sample weights
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))
        elif self.weights is None:
            raise ValueError("Module has been frozen with undefined weights.")

        # Perform embedding lookup
        outputs = self.weights.value[inputs]

        # Apply padding_idx
        if self.padding_idx is not None:
            # Create mask for padding indices
            mask = inputs == self.padding_idx
            # Zero out embeddings for padding indices
            outputs = jnp.where(mask[..., None], 0.0, outputs)

        # Apply max_norm
        if self.max_norm is not None:
            norms = jnp.linalg.norm(outputs, axis=-1, ord=self.norm_type, keepdims=True)
            # Normalize vectors that exceed max_norm
            scale = jnp.minimum(1.0, self.max_norm / (norms + 1e-8))
            outputs = outputs * scale

        return outputs
