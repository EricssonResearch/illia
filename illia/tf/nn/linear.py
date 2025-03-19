# standard libraries
from typing import Optional

# 3pp
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable  # type: ignore

# own modules
from . import (
    Distribution,
    GaussianDistribution,
    BayesianModule,
)


@register_keras_serializable(package="BayesianModule", name="Linear")
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
                (input_size, output_size), name="weights_distr"
            )
        else:
            self.weights_distribution = weights_distribution

        # set bias distribution
        if bias_distribution is None:
            self.bias_distribution: Distribution = GaussianDistribution(
                (output_size,), name="bias_distr"
            )
        else:
            self.bias_distribution = bias_distribution

        # register non-trainable variables
        self.kernel = self.add_weight(
            name="kernel",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape=(input_size, output_size),
            trainable=False,
        )
        self.bias = self.add_weight(
            name="bias",
            initializer=tf.constant_initializer(
                self.bias_distribution.sample().numpy()
            ),
            shape=(output_size,),
            trainable=False,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
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
            self.kernel.assign(self.weights_distribution.sample())
            self.bias.assign(self.bias_distribution.sample())

        else:
            if self.kernel is None or self.bias is None:
                raise ValueError("Module has been frozen with undefined weights")

        # compute outputs
        lin_output = tf.linalg.matmul(inputs, self.kernel)
        outputs: tf.Tensor = tf.nn.bias_add(lin_output, self.bias)
        return outputs

    @tf.function
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        This method is to compute the kl cost of the library.

        Returns:
            kl cost. Dimensions: [].
            number of parameters of the layer.
        """

        log_posterior: tf.Tensor = self.weights_distribution.log_prob(
            self.kernel
        ) + self.bias_distribution.log_prob(self.bias)

        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )
        return log_posterior, num_params

    # def get_config():
    #     super.get_config()

    # @classmethod
    # def from_config():
    #     return
