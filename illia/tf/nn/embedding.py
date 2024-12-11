# Libraries
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

from . import (
    StaticDistribution,
    DynamicDistribution,
    StaticGaussianDistribution,
    DynamicGaussianDistribution,
    BayesianModule,
)


@register_keras_serializable(package="Embedding")
class Embedding(BayesianModule):
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
        Definition of a Bayesian Embedding layer.

        Args:
            num_embeddings: Size of the dictionary of embeddings.
            embeddings_dim: The size of each embedding vector
            weights_prior: The prior distribution for the weights.
            weights_posterior: The posterior distribution for the weights.
            padding_idx: If padding_idx is specified, its entries do not affect the gradient, meaning the
                            embedding vector at padding_idx stays constant during training. Initially, this
                            embedding vector defaults to zeros but can be set to a different value to serve
                            as the padding vector.
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized to have
                            norm max_norm.
            norm_type: The p of the p-norm to compute for the max_norm option.
            scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of the words in the mini-batch.
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.
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

    @tf.fuction
    def _embedding(
        self,
        input: tf.Tensor,
        weight: tf.Tensor,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = 2.0,
        sparse: bool = False,
    ) -> tf.Tensor:
        if sparse is not None:
            embeddings = tf.nn.embedding_lookup(input, weight)
        else:
            embeddings = tf.nn.embedding_lookup_sparse(input, weight)

        if padding_idx is not None:
            padding_mask = tf.not_equal(input, padding_idx)
            embeddings = tf.where(
                tf.expand_dims(padding_mask, -1), embeddings, tf.zeros_like(embeddings)
            )

        if max_norm is not None:
            norms = tf.norm(embeddings, ord=norm_type, axis=-1, keepdims=True)
            desired = tf.clip_by_value(norms, clip_value_min=0, clip_value_max=max_norm)
            scale = desired / (tf.maximum(norms, 1e-7))
            embeddings = embeddings * scale

        return embeddings

    def get_config(self):
        """
        Get the configuration of the Gaussian Distribution object. This method retrieves the base
        configuration of the parent class and combines it with custom configurations specific to
        the Gaussian Distribution.

        Returns:
            A dictionary containing the combined configuration of the Gaussian Distribution.
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
        # Forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()
        else:
            if self.weights is None or self.bias is None:
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
        Calculate the Kullback-Leibler (KL) divergence cost for the weights and bias of the layer.

        Returns:
            A tuple containing the KL divergence cost for the weights and bias, and the total number of parameters.
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
