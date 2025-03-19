# Libraries
from illia.torch.nn.base import BayesianModule as TorchBayesianModule  # noqa
from illia.tf.nn.base import BayesianModule as TFBayesianModule  # noqa
from illia.tf.distributions.static.gaussian import (  # noqa
    GaussianDistribution as TFStaticGaussianDistribution,
)
from illia.torch.distributions.static.gaussian import (  # noqa
    GaussianDistribution as TorchStaticGaussianDistribution,
)

from illia.tf.distributions.dynamic.gaussian import (  # noqa
    GaussianDistribution as TFDynamicGaussianDistribution,
)
from illia.torch.distributions.dynamic.gaussian import (  # noqa
    GaussianDistribution as TorchDynamicGaussianDistribution,
)
