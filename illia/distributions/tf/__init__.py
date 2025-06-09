"""
This module defines the imports for distribution functions in Tensorflow.
"""

from illia.distributions.tf.base import DistributionModule
from illia.distributions.tf.gaussian import GaussianDistribution

# Define all names to be imported
__all__: list[str] = ["DistributionModule", "GaussianDistribution"]
