# Standard libraries
from typing import Any, Optional

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.distributions.torch.gaussian import GaussianDistribution
from illia.nn.torch.base import BayesianModule


class Linear(BayesianModule):
    """
    This class is the bayesian implementation of the torch Linear layer.
    """

    weights: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Linear layer.

        Args:
            input_size: Input size of the linear layer.
            output_size: Output size of the linear layer.
            weights_distribution: GaussianDistribution for the weights of the
                layer. Defaults to None.
            bias_distribution: GaussianDistribution for the bias of the layer.
                Defaults to None.
            use_bias: Whether to include a bias term.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        super().__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                (self.output_size, self.input_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if self.use_bias:
            if bias_distribution is None:
                self.bias_distribution = GaussianDistribution((self.output_size,))
            else:
                self.bias_distribution = bias_distribution
        else:
            self.bias_distribution = None  # type: ignore

        # Sample initial weights
        weights = self.weights_distribution.sample()

        # Register buffers
        self.register_buffer("weights", weights)

        if self.use_bias and self.bias_distribution is not None:
            bias = self.bias_distribution.sample()
            self.register_buffer("bias", bias)
        else:
            self.bias = None  # type: ignore

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:
            self.weights = self.weights_distribution.sample()

        # Sample bias if they are undefined and bias is used
        if self.use_bias and self.bias_distribution is not None:
            if not hasattr(self, "bias") or self.bias is None:
                self.bias = self.bias_distribution.sample()
            self.bias = self.bias.detach()

        # Detach weights and bias
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

        # Compute log probs
        log_probs: torch.Tensor = self.weights_distribution.log_prob(self.weights)

        # Add bias log probs if bias is used
        if self.use_bias and self.bias_distribution is not None:
            log_probs += self.bias_distribution.log_prob(self.bias)

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params()
        if self.use_bias and self.bias_distribution is not None:
            num_params += self.bias_distribution.num_params()

        return log_probs, num_params

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: input tensor. Dimensions: [batch, *].

        Returns:
            outputs tensor. Dimensions: [batch, *].

        Raises:
            ValueError: If the layer is frozen but weights or bias are
                undefined.
        """

        # Check if layer is frozen
        if not self.frozen:
            self.weights = self.weights_distribution.sample()

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution is not None:
                self.bias = self.bias_distribution.sample()
        elif self.weights is None or (self.use_bias and self.bias is None):
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute outputs
        # pylint: disable=E1102
        outputs: torch.Tensor = F.linear(input=inputs, weight=self.weights)

        # Add bias only if using bias
        if self.use_bias and self.bias is not None:
            outputs += torch.reshape(input=self.bias, shape=(1, self.output_size))

        return outputs
