"""
This module contains the code for Embedding Bayesian layer.
"""

from typing import Any, Optional

import tensorflow as tf
from keras import saving

from illia.distributions.tf import GaussianDistribution
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="BayesianModule", name="Embedding")
class Embedding(BayesianModule):
    """
    This class is the bayesian implementation of the Embedding class.
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
        **kwargs: Any,
    ) -> None:
        """
        This method is the constructor of the embedding class.

        Args:
            num_embeddings: Size of the dictionary of embeddings.
            embeddings_dim: The size of each embedding vector.
            padding_idx: If specified, the entries at padding_idx do
                not contribute to the gradient.
            max_norm: If given, each embedding vector with norm larger
                than max_norm is renormalized to have norm max_norm.
            norm_type: The p of the p-norm to compute for the max_norm
                option.
            scale_grad_by_freq: If given, this will scale gradients by
                the inverse of frequency of the words in the
                mini-batch.
            sparse: If True, gradient w.r.t. weight matrix will be a
                sparse tensor.
            weights_distribution: The Gaussian distribution for the
                weights, if applicable.
            **kwargs: Additional keyword arguments.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set atributtes
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_distribution = weights_distribution

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the Embedding layer.

        Args:
            input_shape: Input shape of the layer.
        """

        # Create a variable for weights
        self.w = self.add_weight(
            name="weights",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape=(self.num_embeddings, self.embeddings_dim),
            trainable=False,
        )

        # Call super-class build method
        super().build(input_shape)

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
            "padding_idx": self.padding_idx,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
            "scale_grad_by_freq": self.scale_grad_by_freq,
            "sparse": self.sparse,
        }

        # Combine both configurations
        return {**base_config, **config}

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

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.w is None:
            self.w = self.weights_distribution.sample()

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights.

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
        """
        Performs a forward pass through the Bayesian Embedding layer.

        Samples weights from their posterior distributions if
        the layer is not frozen. If frozen and not initialized, samples
        them once.

        Args:
            inputs: input tensor. Dimensions: [batch, *].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            Output tensor after linear transformation.
        """

        # Check if layer is frozen
        if not self.frozen:
            self.w = self.weights_distribution.sample()
        elif self.w is None:
            raise ValueError("Module has been frozen with undefined weights.")

        # Compute outputs
        outputs: tf.Tensor = self._embedding(
            inputs,
            self.w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.sparse,
        )

        return outputs
