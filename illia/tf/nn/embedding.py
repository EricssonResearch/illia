"""
This module contains the code for Embedding Bayesian layer.
"""

# Standard libraries
from typing import Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.tf.nn.base import BayesianModule
from illia.tf.distributions import GaussianDistribution


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
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        weights_distribution: Optional[GaussianDistribution] = None,
    ) -> None:
        """
        This method is the constructor of the embedding class.

        Args:
            num_embeddings: size of the dictionary of embeddings.
            embeddings_dim: the size of each embedding vector.
            weights_distribution: distribution for the weights of the
                layer. Defaults to None.
            padding_idx: If specified, the entries at padding_idx do
                not contribute to the gradient. Defaults to None.
            max_norm: If given, each embedding vector with norm larger
                than max_norm is renormalized to have norm max_norm.
                Defaults to None.
            norm_type: The p of the p-norm to compute for the max_norm
                option. Defaults to 2.0.
            scale_grad_by_freq: If given, this will scale gradients by
                the inverse of frequency of the words in the
                mini-batch. Defaults to False.
            sparse: If True, gradient w.r.t. weight matrix will be a
                sparse tensor. Defaults to False.
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

        # Set weights distribution
        self.weights_distribution: GaussianDistribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_distribution = weights_distribution

        # Create a variable for weights
        self.w: tf.Variable = self.add_weight(
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            trainable=False,
            name="weights",
            shape=(self.num_embeddings, self.embeddings_dim),
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
            embeddings = tf.nn.embedding_lookup_sparse(weight, inputs, sp_weights=None)

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
        config = {
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
        return {**base_config, **config}

    def freeze(self) -> None:
        """
        This method freezes the layer.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.w is None:
            self.w.assign(self.weights_distribution.sample())  # type: ignore

        # Detach weights stopping training updates
        self.w.assign(tf.stop_gradient(self.w))

    @tf.function
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the KL divergence cost for the layer's weights and
        bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Get log probs
        log_probs: tf.Tensor = self.weights_distribution.log_prob(self.w)

        # Get number of parameters
        num_params: int = self.weights_distribution.num_params

        return log_probs, num_params

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Forward depeding of frozen state
        if not self.frozen:
            self.w.assign(self.weights_distribution.sample())
        else:
            raise ValueError("Module has been frozen with undefined weights")

        # Run tensorflow forward
        outputs: tf.Tensor = self._embedding(
            inputs,
            self.w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.sparse,
        )

        return outputs
