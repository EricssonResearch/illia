# own modules
from illia.tf.distributions.base import Distribution
from illia.tf.distributions.gaussian import GaussianDistribution

# define all names to vbe imported
__all__: list[str] = ["Distribution", "GaussianDistribution"]
