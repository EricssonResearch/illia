# Libraries
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable  # type: ignore

from . import (
    Distribution,
    GaussianDistribution,
    BayesianModule,
)


@register_keras_serializable(package="Embedding")
class Embedding(BayesianModule):
    input_size: int
    output_size: int
    weights_distribution: Distribution
    weights: tf.Tensor
    bias: tf.Tensor

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

        weights_distribution_shape = (num_embeddings, embeddings_dim)
        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                weights_distribution_shape, name="weights_distr"
            )
        else:
            assert (
                weights_distribution.sample().shape == weights_distribution_shape
            ), f"""Expected shape  {weights_distribution_shape}, sampled shape {weights_distribution.sample().shape}"""
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
        input: tf.Tensor,
        weight: tf.Tensor,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = 2.0,
        sparse: bool = False,
    ) -> tf.Tensor:
        input = tf.cast(input, tf.int32)
        if sparse is not None:
            embeddings = tf.nn.embedding_lookup(weight, input)
        else:
            embeddings = tf.nn.embedding_lookup_sparse(weight, input)

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

    def get_config(self) -> dict:
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

        log_posterior: tf.Tensor = self.weights_distribution.log_prob(self.kernel)

        num_params: int = self.weights_distribution.num_params
        return log_posterior, num_params
