"""
This module contains the code for bayesian Embedding layer.
"""

# Standard libraries
from typing import Optional, Any

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.torch.nn.base import BayesianModule
from illia.torch.distributions import GaussianDistribution


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
        Initializes a Bayesian Embedding layer with specified dimensions
        and distributions.

        Args:
            num_embeddings: Number of unique embeddings.
            embeddings_dim: Dimension of each embedding vector.
            weights_distribution: GaussianDistribution for the weights of the
                layer.
            padding_idx: Index for padding, which keeps gradient
                constant.
            max_norm: Maximum norm for embedding vectors.
            norm_type: Norm type for max_norm computation.
            scale_grad_by_freq: Scale gradients by word frequency.
            sparse: Use sparse tensor for weight gradients.
        """

        # Call super class constructor
        super().__init__()

        # Set embeddings atributtes
        self.embedding_params: tuple[Any, ...] = (
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        )

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_distribution = weights_distribution

        # Sample initial weights
        weights = self.weights_distribution.sample()

        # Sample initial weights and register buffers
        self.register_buffer("weights", weights)

    @torch.jit.export
    def freeze(self) -> None:
        """
        This method freezes the layer.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:  # type: ignore
            self.weights = self.weights_distribution.sample()  # pylint: disable=W0201

        # Detach weights
        self.weights = self.weights.detach()  # pylint: disable=W0201

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Get log posterior and log prior
        log_probs: torch.Tensor = self.weights_distribution.log_prob(self.weights)

        # Get number of parameters
        num_params: int = self.weights_distribution.num_params()

        return log_probs, num_params

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Bayesian Embedding layer.

        Samples weights and bias from their posterior distributions if
        the layer is not frozen. If frozen and not initialized, samples
        them once.

        Args:
            inputs: input tensor. Dimensions: [*].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            Output tensor after embedding lookup.
        """

        # forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()  # pylint: disable=W0201
        elif self.weights is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Run torch forward
        outputs: torch.Tensor = F.embedding(
            inputs, self.weights, *self.embedding_params
        )

        return outputs
