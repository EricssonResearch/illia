from typing import Optional, Tuple

import tensorflow as tf
from keras import saving

from . import (
    Distribution,
    GaussianDistribution,
    BayesianModule,
)


@saving.register_keras_serializable(package="BayesianModule", name="Embedding")
class Embedding(BayesianModule):
    """
    Bayesian Embedding layer with trainable weights and biases,
    supporting prior and posterior distributions.
    """

    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        weights_distribution: Optional[Distribution] = None,
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

        # Set atributtes
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        weights_distribution_shape = (num_embeddings, embeddings_dim)
        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                weights_distribution_shape, name="weights_distr"
            )
        else:
            assert weights_distribution.sample().shape == weights_distribution_shape, (
                f"Expected shape  {weights_distribution_shape}, "
                f"sampled shape {weights_distribution.sample().shape}"
            )
            self.weights_distribution = weights_distribution

        # Sample initial distributions
        self.kernel = self.add_weight(
            name="kernel",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape=weights_distribution_shape,
            trainable=False,
        )

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

        inputs = tf.cast(inputs, tf.int32)
        if sparse is not None:
            embeddings = tf.nn.embedding_lookup(weight, inputs)
        else:
            embeddings = tf.nn.embedding_lookup_sparse(weight, inputs)

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
            "weights_distribution": self.weights_distribution,
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
            self.kernel.assign(self.weights_distribution.sample())
        else:
            if self.kernel is None:
                w = self.weights_distribution.sample()
                self.kernel = self.add_weight(
                    name="kernel",
                    initializer=tf.constant_initializer(w.numpy()),
                    shape=w.shape,
                    trainable=False,
                )

        # Run tensorflow forward
        outputs: tf.Tensor = self._embedding(
            inputs,
            self.kernel,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
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

        log_posterior: tf.Tensor = self.weights_distribution.log_prob(self.kernel)

        num_params: int = self.weights_distribution.num_params

        return log_posterior, num_params
