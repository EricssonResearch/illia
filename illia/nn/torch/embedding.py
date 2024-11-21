# Libraries
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import illia.distributions.static as static
import illia.distributions.dynamic as dynamic
from illia.nn.torch.base import BayesianModule


class Embedding(BayesianModule):

    input_size: int
    output_size: int
    weights_posterior: dynamic.DynamicDistribution
    weights_prior: static.StaticDistribution
    bias_posterior: dynamic.DynamicDistribution
    bias_prior: static.StaticDistribution
    weights: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        weights_prior: Optional[static.StaticDistribution] = None,
        weights_posterior: Optional[dynamic.DynamicDistribution] = None,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        """
        Definition of a Bayesian Embedding layer.

        Args:
            num_embeddings (int): Size of the dictionary of embeddings.
            embeddings_dim (int): The size of each embedding vector
            weights_prior (Optional[StaticDistribution], optional): The prior distribution for the weights. Defaults to None.
            weights_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the weights. Defaults to None.
            padding_idx (Optional[int], optional): If padding_idx is specified, its entries do not affect the gradient, meaning the
                                                    embedding vector at padding_idx stays constant during training. Initially, this
                                                    embedding vector defaults to zeros but can be set to a different value to serve
                                                    as the padding vector.
            max_norm (Optional[float], optional): If given, each embedding vector with norm larger than max_norm is renormalized to have
                                                    norm max_norm. Defaults to None.
            norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Defaults to 2.0.
            scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of the words in the
                                                    mini-batch. Defaults to False.
            sparse (bool, optional): If True, gradient w.r.t. weight matrix will be a sparse tensor. Defaults to False.
        """

        # Call super class constructor
        super().__init__()

        # Define default parameters
        parameters = {"mean": 0, "std": 0.1}

        # Set embeddings atributtes
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Set prior if they are None
        if weights_prior is None:
            self.weights_prior = static.GaussianDistribution(
                mu=parameters["mean"], std=parameters["std"]
            )
        else:
            self.weights_prior = weights_prior

        if weights_posterior is None:
            self.weights_posterior = dynamic.GaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_posterior = weights_posterior

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Bayesian Embedding layer.

        If the layer is not frozen, it samples weights and bias from their respective posterior distributions.
        If the layer is frozen and the weights or bias are not initialized, it samples them from their respective posterior distributions.

        Args:
            inputs (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: Output tensor after passing through the layer.
        """

        # Forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()
        else:
            if self.weights is None or self.bias is None:
                self.weights = self.weights_posterior.sample()
                self.bias = self.bias_posterior.sample()

        # Run torch forward
        return F.embedding(
            inputs,
            self.weights,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def kl_cost(self) -> Tuple[torch.Tensor, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the weights and bias of the layer.

        Args:
            self (Conv2d): The instance of the Bayesian Convolution 2D layer.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the KL divergence cost for the weights and bias, and the total number of parameters.
        """

        # Get log posterior and log prior
        log_posterior: torch.Tensor = self.weights_posterior.log_prob(
            self.weights
        ) + self.bias_posterior.log_prob(self.bias)
        log_prior: torch.Tensor = self.weights_prior.log_prob(
            self.weights
        ) + self.bias_prior.log_prob(self.bias)

        # Get number of parameters
        num_params: int = (
            self.weights_posterior.num_params + self.bias_posterior.num_params
        )

        return log_posterior - log_prior, num_params
