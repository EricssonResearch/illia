"""
This module contains the code for bayesian Embedding layer.
"""

# Standard libraries
from typing import Optional, Any

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.nn.torch.base import BayesianModule
from illia.distributions.torch import GaussianDistribution


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
    ) -> None:
        """
        This method is the constructor of the embedding class.

        Args:
            num_embeddings: size of the dictionary of embeddings.
            embeddings_dim: the size of each embedding vector.
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
            weights_distribution: distribution for the weights of the
                layer.
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

        # Register buffers
        self.register_buffer("weights", weights)

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # set indicator
        self.frozen = True

        # sample weights if they are undefined
        if self.weights is None:  # type: ignore
            self.weights = self.weights_distribution.sample()

        # detach weights
        self.weights = self.weights.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # get log posterior and log prior
        log_probs: torch.Tensor = self.weights_distribution.log_prob(self.weights)

        # get number of parameters
        num_params: int = self.weights_distribution.num_params()

        return log_probs, num_params

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: input tensor. Dimensions: [*].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            outputs tensor. Dimension: [*, embedding dim].
        """

        # Forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()
        elif self.weights is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Run torch forward
        outputs: torch.Tensor = F.embedding(
            inputs, self.weights, *self.embedding_params  # type: ignore
        )

        return outputs
