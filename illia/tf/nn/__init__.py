# Libraries
from illia.tf.distributions.static.base import StaticDistribution
from illia.tf.distributions.dynamic.base import DynamicDistribution
from illia.tf.distributions.static.gaussian import (
    GaussianDistribution as StaticGaussianDistribution,
)
from illia.tf.distributions.dynamic.gaussian import (
    GaussianDistribution as DynamicGaussianDistribution,
)
from illia.tf.nn.base import BayesianModule
