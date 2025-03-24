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
    This class is the bayesian implementation of the Embedding class.

    Attr:
        weights_distribution: distribution for the weights of the
            layer. Dimensions: [number of embeddings, embedding dim].
        weights: sampled weights of the layer. They are registered in
            the buffer. Dimensions: [number of embeddings,
            embedding dim].
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

        # call super class constructor
        super().__init__()

        # Set embeddings atributtes
        self.embedding_params: tuple[Any, ...] = (
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        )

        # set weights distribution
        self.weights_distribution: GaussianDistribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_distribution = weights_distribution

        # sample initial weights
        weights = self.weights_distribution.sample()

        # register buffers
        self.register_buffer("weights", weights)

        return None

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

        # forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()  # pylint: disable=W0201

        else:
            if self.weights is None:
                raise ValueError("Module has been frozen with undefined weights")

        # run torch forward
        outputs: torch.Tensor = F.embedding(  # # pylint: disable=E1102
            inputs, self.weights, *self.embedding_params  # type: ignore
        )

        return outputs

    @torch.jit.export
    def freeze(self) -> None:
        """
        This method freezes the layer.

        Returns:
            None.
        """

        # set indicator
        self.frozen = True

        # sample weights if they are undefined
        if self.weights is None:
            self.weights = self.weights_distribution.sample()  # pylint: disable=W0201

        # detach weights
        self.weights = self.weights.detach()  # pylint: disable=W0201

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        This method calculates the kl cost of the layer.

        Returns:
            kl cost. Dimensions: [].
            number of parameters of the layer. It can be used to
                average the kl cost.
        """

        # get log posterior and log prior
        log_probs: torch.Tensor = self.weights_distribution.log_prob(self.weights)

        # get number of parameters
        num_params: int = self.weights_distribution.num_params()

        return log_probs, num_params
