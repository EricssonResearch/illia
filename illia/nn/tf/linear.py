# Libraries
from typing import Optional, Tuple, Union

import tensorflow as tf

import illia.distributions.static as static
import illia.distributions.dynamic as dynamic
from illia.nn.base import BayesianModule


class Linear(BayesianModule):

    input_size: int
    output_size: int
    weights_posterior: dynamic.DynamicDistribution
    weights_prior: static.StaticDistribution
    bias_posterior: dynamic.DynamicDistribution
    bias_prior: static.StaticDistribution
    weights: tf.Tensor
    bias: tf.Tensor

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
            input_size: Size of each input sample.
            output_size: Size of each output sample.
            weights_prior: The prior distribution for the weights.
            bias_prior: The prior distribution for the bias.
            weights_posterior: The posterior distribution for the weights.
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
            self.weights_prior = static.GaussianDistribution(
                parameters["mean"], parameters["std"]
            )
        else:
            self.weights_prior = weights_prior

        # Set bias prior
        if bias_prior is None:
            self.bias_prior = static.GaussianDistribution(
                parameters["mean"], parameters["std"]
            )
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

    def get_config(self):
        """
        Get the configuration of the Gaussian Distribution object. This method retrieves the base
        configuration of the parent class and combines it with custom configurations specific to
        the Gaussian Distribution.

        Returns:
            A dictionary containing the combined configuration of the Gaussian Distribution.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "num_embeddings": self.num_embeddings,
            "embeddings_dim": self.embeddings_dim,
            "weights_prior": self.weights_prior,
            "weights_posterior": self.weights_posterior,
            "padding_idx": self.padding_idx,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
            "scale_grad_by_freq": self.scale_grad_by_freq,
            "sparse": self.sparse,
        }

        # Combine both configurations
        return {**base_config, **custom_config}
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs a forward pass through the Bayesian Linear layer.

        If the layer is not frozen, it samples weights and bias from their respective posterior distributions.
        If the layer is frozen and the weights or bias are not initialized, it samples them from their respective posterior distributions.

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

        # Run tf forward
        return tf.linalg.matmul(inputs, self.weights) + self.bias

    @tf.function
    def kl_cost(self) -> Tuple[tf.Tensor, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the weights and bias of the layer.

        Returns:
            A tuple containing the KL divergence cost for the weights and bias, and the total number of parameters.
        """

        log_posterior: tf.Tensor = self.weights_posterior.log_prob(
            self.weights
        ) + self.bias_posterior.log_prob(self.bias)
        log_prior: tf.Tensor = self.weights_prior.log_prob(
            self.weights
        ) + self.bias_prior.log_prob(self.bias)

        num_params: int = (
            self.weights_posterior.num_params + self.bias_posterior.num_params
        )

        return log_posterior - log_prior, num_params
