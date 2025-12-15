# Standard libraries
from typing import Any, Optional

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.distributions.torch.gaussian import GaussianDistribution
from illia.nn.torch.base import BayesianModule


class Embedding(BayesianModule):
    """
    This class is the bayesian implementation of the Embedding class.
    """

    weights: torch.Tensor

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
        Initializes a Embedding layer.

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
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        super().__init__(**kwargs)

        # Set embeddings atributtes
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
                (self.num_embeddings, self.embeddings_dim)
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
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

        Returns:
            None.
        """

        # set indicator
        self.frozen = True

        # sample weights if they are undefined
        if self.weights is None:
            self.weights = self.weights_distribution.sample()

        # detach weights
        self.weights = self.weights.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the KL
                divergence cost and the total number of parameters in
                the layer.
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

        Returns:
            outputs tensor. Dimension: [*, embedding dim].

        Raises:
            ValueError: If the layer is frozen but weights are
                undefined.
        """

        # Forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()
        elif self.weights is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Run torch forward
        return F.embedding(
            input=inputs,
            weight=self.weights,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
