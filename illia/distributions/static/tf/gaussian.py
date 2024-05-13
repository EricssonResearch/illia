# deep larning libraries
import tensorflow as tf

# other libraries
from typing import Dict

# own modules
from illia.distributions.static import StaticDistribution

# static variables
PI: tf.Tensor = tf.math.acos(tf.zeros(1)) * 2


class GaussianDistribution(StaticDistribution):
    
    def __init__(self, mu: float, std: float) -> None:
        """
        This method is the constrcutor of the class.

        Args:
            mu: mu parameter.
            std: standard deviation parameter.
        """
        
        # set attributes
        self.mu = tf.Tensor(mu)
        self.std = tf.Tensor(std)

    # overriding method
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        """
        This method computes the log probabilities.

        Args:
            x: _description_

        Returns:
            output tensor. Dimensions: 
        """
        
        # change device
        self.mu = self.mu.to(x.device)
        self.std = self.std.to(x.device)

        # compute log probs
        log_probs = (
            -tf.log(tf.sqrt(2 * PI)).to(x.device)
            - tf.log(self.std)
            - (((x - self.mu) ** 2) / (2 * self.std**2))
            - 0.5
        )

        return log_probs.sum()
