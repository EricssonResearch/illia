# Libraries
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import illia.distributions.static as static
import illia.distributions.dynamic as dynamic
from illia.nn.torch.base import BayesianModule


class Linear(BayesianModule):
    """
    This class is the Linear bayesian layer.

    Attr:
        input_size: input size of the Linear Layer.
        output_size: output size of the Linear layer.
        weights_posterior:

    Returns:
        _description_
    """

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
        This methos is the forward pass of the model.

        Args:
            inputs: inputs of the model. Dimensions: [*, input size].

        Returns:
            output tensor. Dimension: [*, output size].
        """

        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()

        else:
            if self.weights is None or self.bias is None:
                self.weights = self.weights_posterior.sample()
                self.bias = self.bias_posterior.sample()

        # compurte the outputs
        outputs: torch.Tensor = F.linear(inputs, self.weights, self.bias)

        return outputs

    def kl_cost(self) -> Tuple[torch.Tensor, int]:
        """
        This method computes the kl-divergence cost for the layer.

        Returns:
            kl cost.
            number of parameters of the layer.
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
