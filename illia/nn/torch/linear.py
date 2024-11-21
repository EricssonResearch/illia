# Libraries
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import illia.distributions.static as static
import illia.distributions.dynamic as dynamic
from illia.nn.torch.base import BayesianModule


class Linear(BayesianModule):

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
        input_size: int,
        output_size: int,
        weights_prior: Optional[static.StaticDistribution] = None,
        bias_prior: Optional[static.StaticDistribution] = None,
        weights_posterior: Optional[dynamic.DynamicDistribution] = None,
        bias_posterior: Optional[dynamic.DynamicDistribution] = None,
    ) -> None:
        """
        Definition of a Bayesian Linear layer.

        Args:
            input_size (int): Size of each input sample.
            output_size (int): Size of each output sample.
            weights_prior (Optional[StaticDistribution], optional): The prior distribution for the weights. Defaults to None.
            bias_prior (Optional[StaticDistribution], optional): The prior distribution for the bias. Defaults to None.
            weights_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the weights. Defaults to None.
            bias_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the bias. Defaults to None.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.input_size = input_size
        self.output_size = output_size

        # Set defaults parameters for gaussian
        mean: float = 0.0
        std: float = 0.1

        # Set weights prior
        if weights_prior is None:
            self.weights_prior = static.GaussianDistribution(mean, std)
        else:
            self.weights_prior = weights_prior

        # Set bias prior
        if bias_prior is None:
            self.bias_prior = static.GaussianDistribution(mean, std)
        else:
            self.bias_prior = bias_prior

        # Set weights posterior
        if weights_posterior is None:
            self.weights_posterior = dynamic.GaussianDistribution(
                (output_size, input_size)
            )
        else:
            self.weights_posterior = weights_posterior

        # Set bias posterior
        if bias_posterior is None:
            self.bias_posterior = dynamic.GaussianDistribution((output_size,))
        else:
            self.bias_posterior = bias_posterior

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Bayesian Linear layer.

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
        return F.linear(inputs, self.weights, self.bias)

    def kl_cost(self) -> Tuple[torch.Tensor, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the weights and bias of the layer.

        Args:
            self (Conv2d): The instance of the Bayesian Convolution 2D layer.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the KL divergence cost for the weights and bias, and the total number of parameters.
        """

        log_posterior: torch.Tensor = self.weights_posterior.log_prob(
            self.weights
        ) + self.bias_posterior.log_prob(self.bias)
        log_prior: torch.Tensor = self.weights_prior.log_prob(
            self.weights
        ) + self.bias_prior.log_prob(self.bias)

        num_params: int = (
            self.weights_posterior.num_params + self.bias_posterior.num_params
        )

        return log_posterior - log_prior, num_params
