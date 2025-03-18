# Libraries
from typing import Optional

import torch
import torch.nn.functional as F

from . import (
    StaticDistribution,
    DynamicDistribution,
    StaticGaussianDistribution,
    DynamicGaussianDistribution,
    BayesianModule,
)


class Linear(BayesianModule):
    """
    Represents a Bayesian Linear layer that models uncertainty in its
    weights and biases using prior and posterior distributions.
    """

    input_size: int
    output_size: int
    weights_posterior: DynamicDistribution
    weights_prior: StaticDistribution
    bias_posterior: DynamicDistribution
    bias_prior: StaticDistribution
    weights: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_prior: Optional[StaticDistribution] = None,
        bias_prior: Optional[StaticDistribution] = None,
        weights_posterior: Optional[DynamicDistribution] = None,
        bias_posterior: Optional[DynamicDistribution] = None,
    ) -> None:
        """
        Definition of a Bayesian Linear layer.

        Args:
            input_size: Size of each input sample.
            output_size: Size of each output sample.
            weights_prior: The prior distribution for the weights.
            bias_prior: The prior distribution for the bias.
            weights_posterior: The posterior distribution for the
                weights.
            bias_posterior: The posterior distribution for the bias.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.input_size = input_size
        self.output_size = output_size

        # Define default parameters
        parameters = {"mean": 0, "std": 0.1}

        # Set weights prior
        if weights_prior is None:
            self.weights_prior = StaticGaussianDistribution(
                parameters["mean"], parameters["std"]
            )
        else:
            self.weights_prior = weights_prior

        # Set bias prior
        if bias_prior is None:
            self.bias_prior = StaticGaussianDistribution(
                parameters["mean"], parameters["std"]
            )
        else:
            self.bias_prior = bias_prior

        # Set weights posterior
        if weights_posterior is None:
            self.weights_posterior = DynamicGaussianDistribution(
                (output_size, input_size)
            )
        else:
            self.weights_posterior = weights_posterior

        # Set bias posterior
        if bias_posterior is None:
            self.bias_posterior = DynamicGaussianDistribution((output_size,))
        else:
            self.bias_posterior = bias_posterior

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Bayesian Linear layer.

        If the layer is not frozen, it samples weights and bias from
        their respective posterior distributions. If the layer is
        frozen and the weights or bias are not initialized, it samples
        them from their respective posterior distributions.

        Args:
            inputs: Input tensor to the layer.

        Returns:
            Output tensor after passing through the layer.
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

    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the
        weights and bias of the layer.

        Returns:
            A tuple containing the KL divergence cost for the weights
            and bias, and the total number of parameters.
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
