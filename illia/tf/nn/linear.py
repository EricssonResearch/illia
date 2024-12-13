# standard libraries
from typing import Optional

# 3pp
import tensorflow as tf

# own modules
from illia.tf.nn.base import BayesianModule
from illia.tf.distributions import (
    Distribution,
    GaussianDistribution,
)


class Linear(BayesianModule):
    """
    This class is the bayesian implementation of the tensorflow Linear
    layer.

    Attr:
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
        weights_distribution: Optional[Distribution] = None,
        bias_distribution: Optional[Distribution] = None,
    ) -> None:
        """
        This is the constructor of the Linear class.

        Args:
            input_size: input size of the linear layer.
            output_size: output size of the linear layer.
            weights_distribution: distribution for the weights of the
                layer. Defaults to None.
            bias_distribution: distribution for the bias of the layer.
                Defaults to None.
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

        # register non-trainable variables
        self.weights = tf.keras.Variable(weights, trainable=False)
        self.bias = tf.keras.Variable(bias, trainable=False)

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: input tensor. Dimensions: [batch, *].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            outputs tensor. Dimensions: [batch, *].
        """

        # check if layer is frozen
        if not self.frozen:
            self.weights = self.weights_distribution.sample()
            self.bias = self.bias_distribution.sample()

        else:
            if self.weights is None or self.bias is None:
                raise ValueError("Module has been frozen with undefined weights")

        # compute outputs
        outputs: tf.Tensor = tf.linalg.matmul(inputs, self.weights) + self.bias

        return outputs

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
        # self.weights = self.weights.detach()
        # self.bias = self.bias.detach()

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        This method is to compute the kl cost of the library.

        Returns:
            kl cost. Dimensions: [].
            number of parameters of the layer.
        """

        # compute log probs
        log_probs: tf.Tensor = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # compute the number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params
