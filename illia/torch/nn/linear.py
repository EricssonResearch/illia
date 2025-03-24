"""
This module contains the code for Linear Bayesian layer.
"""

# Standard libraries
from typing import Optional

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.torch.nn.base import BayesianModule
from illia.torch.distributions import GaussianDistribution


class Linear(BayesianModule):
    """
    This class is the bayesian implementation of the torch Linear layer.

    Attributes:
        weights_distribution: distribution for the weights of the
            layer. Dimensions: [output size, input size].
        bias_distribution: distribution of the bias layer. Dimensions:
            [output size].
        weights: sampled weights of the layer. They are registered in
            the buffer. Dimensions: [output size, input size].
        bias: sampled bias of the layer. They are registered in
            the buffer. Dimensions: [output size].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
    ) -> None:
        """
        This is the constructor of the Linear class.

        Args:
            input_size: Input size of the linear layer.
            output_size: Output size of the linear layer.
            weights_distribution: GaussianDistribution for the weights of the
                layer. Defaults to None.
            bias_distribution: GaussianDistribution for the bias of the layer.
                Defaults to None.
        """

        # Call super-class constructor
        super().__init__()

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution: GaussianDistribution = GaussianDistribution(
                (output_size, input_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            self.bias_distribution: GaussianDistribution = GaussianDistribution(
                (output_size,)
            )
        else:
            self.bias_distribution = bias_distribution

        # Sample initial weights
        weights = self.weights_distribution.sample()
        bias = self.bias_distribution.sample()

        # Register buffers
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)

    @torch.jit.export
    def freeze(self) -> None:
        """
        This method is to freeze the layer.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:  # type: ignore
            self.weights = self.weights_distribution.sample()  # pylint: disable=W0201

        # Sample bias is they are undefined
        if self.bias is None:  # type: ignore
            self.bias = self.bias_distribution.sample()  # pylint: disable=W0201

        # Detach weights and bias
        self.weights = self.weights.detach()  # pylint: disable=W0201
        self.bias = self.bias.detach()  # pylint: disable=W0201

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: input tensor. Dimensions: [batch, *].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            outputs tensor. Dimensions: [batch, *].
        """

        # Check if layer is frozen
        if not self.frozen:
            self.weights = self.weights_distribution.sample()  # pylint: disable=W0201
            self.bias = self.bias_distribution.sample()  # pylint: disable=W0201
        elif self.weights is None or self.bias is None:
            raise ValueError("Module has been frozen with undefined weights")

        # compute outputs
        outputs: torch.Tensor = F.linear(  # pylint: disable=E1102
            inputs, self.weights, self.bias
        )

        return outputs
