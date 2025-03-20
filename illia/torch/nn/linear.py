"""
This module contains the code for Linear Bayesian layer.
"""

# Standard libraries
from typing import Optional

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.torch.nn import BayesianModule
from illia.torch.distributions import (
    Distribution,
    GaussianDistribution,
)


class Linear(BayesianModule):
    """
    Represents a Bayesian Linear layer that models uncertainty in its
    weights and biases using prior and posterior distributions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[Distribution] = None,
        bias_distribution: Optional[Distribution] = None,
    ) -> None:
        """
        This is the constructor of the Linear class.

        Args:
            input_size: Size of each input sample.
            output_size: Size of each output sample.
            weights_distribution: Distribution for the weights of the
                layer.
            bias_distribution: Distribution for the bias of the layer.
        """

        # Call super-class constructor
        super().__init__()

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                (output_size, input_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            self.bias_distribution: Distribution = GaussianDistribution((output_size,))
        else:
            self.bias_distribution = bias_distribution

        # Sample initial weights
        weights = self.weights_distribution.sample()
        bias = self.bias_distribution.sample()

        # Register buffers
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Bayesian Linear layer.

        If the layer is not frozen, it samples weights and bias from
        their respective posterior distributions. If the layer is
        frozen and the weights or bias are not initialized, it samples
        them from their respective posterior distributions.

        Args:
            inputs: input tensor. Dimensions: [batch, *].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            outputs tensor. Dimensions: [batch, *].
        """

        # Check if layer is frozen
        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()
        elif self.weights is None or self.bias is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Run torch forward
        outputs: torch.Tensor = F.linear(inputs, self.weights, self.bias)

        return outputs

    @torch.jit.export
    def freeze(self) -> None:
        """
        This method is to freeze the layer.

        Returns:
            None.
        """

        # set indicator
        self.frozen = True

        # sample weights if they are undefined
        if self.weights is None:
            self.weights = self.weights_distribution.sample()

        # sample bias is they are undefined
        if self.bias is None:
            self.bias = self.bias_distribution.sample()

        # detach weights and bias
        self.weights = self.weights.detach()
        self.bias = self.bias.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the
        weights and bias of the layer.

        Returns:
            A tuple containing the KL divergence cost for the weights
            and bias, and the total number of parameters.
        """

        # Compute log probs
        log_probs: torch.Tensor = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # Compute the number of parameters
        num_params: int = (
            self.weights_distribution.num_params() + self.bias_distribution.num_params()
        )

        return log_probs, num_params
