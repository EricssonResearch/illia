# Standard libraries
from typing import Any, Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf.gaussian import GaussianDistribution
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="illia", name="Embedding")
class Embedding(BayesianModule):
    """
    Bayesian embedding layer with optional padding and max-norm. Each
    embedding vector is sampled from a specified distribution. Can be
    frozen to fix embeddings and stop gradients.
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
        Initializes a Bayesian Embedding layer.

        Args:
            num_embeddings: Size of the embedding dictionary.
            embeddings_dim: Dimensionality of each embedding vector.
            padding_idx: Index to exclude from gradient computation.
            max_norm: Maximum norm for embedding vectors.
            norm_type: p of the p-norm for max_norm.
            scale_grad_by_freq: Scale gradient by inverse frequency.
            sparse: Use sparse gradient updates.
            weights_distribution: Distribution for embedding weights.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

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
        Build trainable and non-trainable parameters.

        Args:
            input_shape: Input shape used to trigger layer build.

        Returns:
            None
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

        super().build(input_shape)

    def get_config(self) -> dict:
        """
        Return the configuration dictionary for serialization.

        Returns:
            dict: Dictionary with the layer configuration.
        """

        base_config = super().get_config()

        config = {
            "num_embeddings": self.num_embeddings,
            "embeddings_dim": self.embeddings_dim,
            "padding_idx": self.padding_idx,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
            "scale_grad_by_freq": self.scale_grad_by_freq,
            "sparse": self.sparse,
        }

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
        Computes embedding lookup with optional padding and normalization.

        Args:
            inputs: Input tensor of indices.
            weight: Embedding weight tensor.
            padding_idx: Index to mask out.
            max_norm: Maximum norm for embeddings.
            norm_type: Norm type for max_norm.
            sparse: Use sparse lookup if True.

        Returns:
            Tensor of embeddings.
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
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.w is None:
            self.w = self.weights_distribution.sample()

        # Stop gradient computation
        self.w = tf.stop_gradient(self.w)

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[tf.Tensor, int]: A tuple containing the KL divergence
                cost and the total number of parameters in the layer.
        """

        # Get log probs
        log_probs: tf.Tensor = self.weights_distribution.log_prob(self.w)

        # Get number of parameters
        num_params: int = self.weights_distribution.num_params

        return log_probs, num_params

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs embedding lookup using current weights.

        Args:
            inputs: Input tensor of indices with shape [batch, *].

        Returns:
            Tensor of embeddings.

        Raises:
            ValueError: If the layer is frozen but weights are
                undefined.
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
