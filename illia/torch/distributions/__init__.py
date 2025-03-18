# own modules
from illia.torch.distributions.base import Distribution
from illia.torch.distributions.gaussian import GaussianDistribution

# define all names to vbe imported
__all__: list[str] = ["Distribution", "GaussianDistribution"]
