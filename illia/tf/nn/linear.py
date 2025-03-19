# Libraries
from typing import Optional

import tensorflow as tf
from keras import saving

from . import (
    Distribution,
    GaussianDistribution,
    BayesianModule,
)


@saving.register_keras_serializable(package="BayesianModule", name="Linear")
class Linear(BayesianModule):
    """
    Bayesian Linear layer with trainable weights and biases,
    supporting prior and posterior distributions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[Distribution] = None,
        bias_distribution: Optional[Distribution] = None,
    ) -> None:
        """
        Initializes a Bayesian Linear layer with specified dimensions
        and distributions.

        Args:
            input_size: Number of features in input.
            output_size: Number of features in output.
            weights_distribution: distribution for the weights of the
                layer.
            bias_distribution: distribution for the bias of the layer.
        """

        # Call super-class constructor
        super().__init__()

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                 (input_size, output_size),
                 name="weights_distr"
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            self.bias_distribution: Distribution = GaussianDistribution(
                (output_size,),
                name="bias_distr"
            )
        else:
            self.bias_distribution = bias_distribution

        # Register non-trainable variables
        self.kernel = self.add_weight(
            name="kernel",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape= (input_size, output_size),
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

    def get_config(self) -> dict:
        """
        Retrieves the configuration of the Linear layer.

        Returns:
            Dictionary containing layer configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weights_distribution": self.weights_distribution,
            "bias_distribution": self.bias_distribution
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs a forward pass through the Bayesian Linear layer.

        Samples weights and bias from their posterior distributions if
        the layer is not frozen. If frozen and not initialized, samples
        them once.

        Args:
            inputs: Input tensor to the layer.

        Returns:
            Output tensor after linear transformation.
        """

        # Check if layer is frozen
        if not self.frozen:
            self.kernel.assign(self.weights_distribution.sample())
            self.bias.assign(self.bias_distribution.sample())
        elif self.kernel is None or self.bias is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Compute outputs
        lin_output = tf.linalg.matmul(inputs, self.kernel) 
        outputs: tf.Tensor = tf.nn.bias_add(lin_output, self.bias)
        
        return outputs

    @tf.function
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        log_posterior: tf.Tensor = self.weights_distribution.log_prob(
            self.kernel
        ) + self.bias_distribution.log_prob(self.bias)

        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )
        
        return log_posterior, num_params
