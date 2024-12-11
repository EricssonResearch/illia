# other libraries
from typing import Optional

# 3pp
import torch
import torch.nn.functional as F

# own modules
from illia.torch.nn.base import BayesianModule
from illia.torch.distributions import (
    Distribution,
    GaussianDistribution,
)


class Linear(BayesianModule):
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
            input_size: _description_
            output_size: _description_
            weights_distribution: _description_. Defaults to None.
            bias_distribution: _description_. Defaults to None.
        """

        # call super-class constructor
        super().__init__()

        # set weights distribution
        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                (output_size, input_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # set bias distribution
        if bias_distribution is None:
            self.bias_distribution: Distribution = GaussianDistribution((output_size,))
        else:
            self.bias_distribution = bias_distribution

        # sample initial weights
        weights = self.weights_distribution.sample()
        bias = self.bias_distribution.sample()

        # register buffers
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: input tensor. Dimensions: [*].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            _description_
        """

        if not self.frozen:
            self.weights = self.weights_distribution.sample()
            self.bias = self.bias_distribution.sample()

        else:
            if self.weights is None or self.bias is None:
                raise ValueError("Module has been frozen with undefined weights")

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
        # if self.bias

        # detach weights and bias
        self.weights = self.weights.detach()
        self.bias = self.bias.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        This method is to compute the kl cost of the library.

        Returns:
            _description_
        """

        # compute log probs
        log_probs: torch.Tensor = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # compute the number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params
