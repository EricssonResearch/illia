# Libraries
from illia.torch.nn.base import BayesianModule as TorchBayesianModule
from illia.tf.nn.base import BayesianModule as TFBayesianModule
from illia.tf.distributions.static.gaussian import (
    GaussianDistribution as TFStaticGaussianDistribution,
)
from illia.torch.distributions.static.gaussian import (
    GaussianDistribution as TorchStaticGaussianDistribution,
)

from illia.tf.distributions.dynamic.gaussian import (
    GaussianDistribution as TFDynamicGaussianDistribution,
)
from illia.torch.distributions.dynamic.gaussian import (
    GaussianDistribution as TorchDynamicGaussianDistribution,
)
