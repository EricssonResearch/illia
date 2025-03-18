# Libraries
from typing import Optional, Tuple

import tensorflow as tf
from keras import saving

from . import (
    StaticDistribution,
    DynamicDistribution,
    StaticGaussianDistribution,
    DynamicGaussianDistribution,
    BayesianModule,
)


@saving.register_keras_serializable(package="illia_tf", name="Embedding")
class Embedding(BayesianModule):
    """
    Bayesian Embedding layer with trainable weights and biases,
    supporting prior and posterior distributions.
    """

    input_size: int
    output_size: int
    weights_posterior: DynamicDistribution
    weights_prior: StaticDistribution
    bias_posterior: DynamicDistribution
    bias_prior: StaticDistribution
    weights: tf.Tensor
    bias: tf.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        weights_prior: Optional[StaticDistribution] = None,
        weights_posterior: Optional[DynamicDistribution] = None,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        """
        Initializes a Bayesian Embedding layer with specified dimensions
        and distributions.

        Args:
            num_embeddings: Number of unique embeddings.
            embeddings_dim: Dimension of each embedding vector.
            weights_prior: Prior distribution for weights.
            weights_posterior: Posterior distribution for weights.
            padding_idx: Index for padding, which keeps gradient
                constant.
            max_norm: Maximum norm for embedding vectors.
            norm_type: Norm type for max_norm computation.
            scale_grad_by_freq: Scale gradients by word frequency.
            sparse: Use sparse tensor for weight gradients.
        """

        # Call super class constructor
        super().__init__()

        # Define parameters
        parameters = {"mean": 0, "std": 0.1}

        # Set atributtes
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Set prior if they are None
        if weights_prior is None:
            self.weights_prior = StaticGaussianDistribution(
                mu=parameters["mean"], std=parameters["std"]
            )
        else:
            self.weights_prior = weights_prior

        # Set posterior if they are None
        if weights_posterior is None:
            self.weights_posterior = DynamicGaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_posterior = weights_posterior

    @tf.function
    def _embedding(
        self,
        inputs: tf.Tensor,
        weight: tf.Tensor,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = 2.0,
        sparse: bool = False,
    ) -> tf.Tensor:
        """
        Computes the embedding lookup with optional padding and
        normalization.

        Args:
            inputs: Input tensor for lookup.
            weight: Weight tensor for embeddings.
            padding_idx: Index to pad embeddings.
            max_norm: Maximum norm for embeddings.
            norm_type: Norm type for normalization.
            sparse: Use sparse lookup.

        Returns:
            Tensor containing the computed embeddings.
        """

        if sparse is not None:
            embeddings = tf.nn.embedding_lookup(inputs, weight)
        else:
            embeddings = tf.nn.embedding_lookup_sparse(inputs, weight)

        if padding_idx is not None:
            padding_mask = tf.not_equal(inputs, padding_idx)
            embeddings = tf.where(
                tf.expand_dims(padding_mask, -1), embeddings, tf.zeros_like(embeddings)
            )

        if max_norm is not None:
            norms = tf.norm(embeddings, ord=norm_type, axis=-1, keepdims=True)
            desired = tf.clip_by_value(norms, clip_value_min=0, clip_value_max=max_norm)
            scale = desired / (tf.maximum(norms, 1e-7))
            embeddings = embeddings * scale

        return embeddings

    def get_config(self) -> dict:
        """
        Retrieves the configuration of the Embedding layer.

        Returns:
            Dictionary containing layer configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "num_embeddings": self.num_embeddings,
            "embeddings_dim": self.embeddings_dim,
            "weights_prior": self.weights_prior,
            "weights_posterior": self.weights_posterior,
            "padding_idx": self.padding_idx,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
            "scale_grad_by_freq": self.scale_grad_by_freq,
            "sparse": self.sparse,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs forward pass to compute embeddings based on input
        tensor.

        Args:
            inputs: Input tensor for the layer.

        Returns:
            Tensor containing the layer's output.
        """

        # Forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()
        elif self.weights is None or self.bias is None:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()

        # Run tensorflow forward
        outputs: tf.Tensor = self._embedding(
            inputs,
            self.weights,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            # TODO: self.scale_grad_by_freq,
            self.sparse,
        )

        return outputs

    @tf.function
    def kl_cost(self) -> Tuple[tf.Tensor, int]:
        """
        Computes the KL divergence cost for the layer's weights and
        bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Get log posterior and log prior
        log_posterior: tf.Tensor = self.weights_posterior.log_prob(
            self.weights
        ) + self.bias_posterior.log_prob(self.bias)
        log_prior: tf.Tensor = self.weights_prior.log_prob(
            self.weights
        ) + self.bias_prior.log_prob(self.bias)

        # Get number of parameters
        num_params: int = (
            self.weights_posterior.num_params + self.bias_posterior.num_params
        )

        return log_posterior - log_prior, num_params
