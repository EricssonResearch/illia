# Libraries
import random
from typing import Dict, Tuple

import pytest
import numpy as np
import torch
import tensorflow as tf

from illia.distributions.static.gaussian import (
    GaussianDistribution as BackendAgnosticStaticGaussian,
)
from illia.distributions.dynamic.gaussian import (
    GaussianDistribution as BackendAgnosticDynamicGaussian,
)
from illia.nn.torch.base import BayesianModule as TorchBayesianModule
from illia.nn.tf.base import BayesianModule as TFBayesianModule

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


class TorchTestModule(TorchBayesianModule):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def kl_cost(self):
        return torch.tensor(1.0), 1


class TFTestModule(TFBayesianModule):

    def __init__(self):
        super().__init__()
        self.linear = tf.keras.layers.Dense(5, activation=None)

    def call(self, x):
        return self.linear(x)

    def kl_cost(self):
        return tf.constant(1.0), 1


@pytest.fixture
def set_parameters() -> Dict:
    """
    This function sets the parameters for the Gaussian distributions and the Bayesian neural network modules.

    Returns:
        Dict: A dictionary containing the following parameters:
            shape: A tuple representing the shape of the Gaussian distribution.
            mu_prior: A float representing the mean of the prior distribution.
            std_prior: A float representing the standard deviation of the prior distribution.
            mu_init: A float representing the initial mean for the Gaussian distribution.
            rho_init: A float representing the initial rho (log standard deviation) for the Gaussian distribution.
    """

    shape = (3, 2)
    mu_prior = 0.0
    std_prior = 0.1
    mu_init = 0.0
    rho_init = -7.0

    return {
        "shape": shape,
        "mu_prior": mu_prior,
        "std_prior": std_prior,
        "mu_init": mu_init,
        "rho_init": rho_init,
    }


@pytest.fixture
def set_dynamic_distributions(set_parameters) -> Tuple:
    """
    This function initializes dynamic Gaussian distributions for both PyTorch and TensorFlow backends.

    Args:
        set_parameters : Dict
            A dictionary containing the parameters for the Gaussian distributions.
            The dictionary should contain the following keys:
            - "shape" : A tuple representing the shape of the Gaussian distribution.
            - "mu_init" : A float representing the initial mean for the Gaussian distribution.
            - "rho_init" : A float representing the initial rho (log standard deviation) for the Gaussian distribution.

    Returns:
        Tuple
            A tuple containing two initialized dynamic Gaussian distributions:
            - torch_dynamic_dist : A dynamic Gaussian distribution for PyTorch backend.
            - tf_dynamic_dist : A dynamic Gaussian distribution for TensorFlow backend.
    """

    # Access to the variables
    shape = set_parameters["shape"]
    mu_init = set_parameters["mu_init"]
    rho_init = set_parameters["rho_init"]

    # Initialize dynamic distributions
    torch_dynamic_dist = BackendAgnosticDynamicGaussian(
        shape=shape, mu_init=mu_init, rho_init=rho_init, backend="torch"
    )
    tf_dynamic_dist = BackendAgnosticDynamicGaussian(
        shape=shape, mu_init=mu_init, rho_init=rho_init, backend="tf"
    )

    return torch_dynamic_dist, tf_dynamic_dist


@pytest.fixture
def set_static_distributions(set_parameters) -> Tuple:
    """
    This function initializes static Gaussian distributions for both PyTorch and TensorFlow backends.

    Args:
        set_parameters : Dict
            A dictionary containing the parameters for the Gaussian distributions.
            The dictionary should contain the following keys:
            - "mu_prior" : A float representing the mean of the prior distribution.
            - "std_prior" : A float representing the standard deviation of the prior distribution.

    Returns:
        Tuple
            A tuple containing two initialized static Gaussian distributions:
            - torch_static_dist : A static Gaussian distribution for PyTorch backend.
            - tf_static_dist : A static Gaussian distribution for TensorFlow backend.
    """

    # Access to the variables
    mu_prior = set_parameters["mu_prior"]
    std_prior = set_parameters["std_prior"]

    # Initialize static distributions
    torch_static_dist = BackendAgnosticStaticGaussian(
        mu=mu_prior, std=std_prior, backend="torch"
    )
    tf_static_dist = BackendAgnosticStaticGaussian(
        mu=mu_prior, std=std_prior, backend="tf"
    )

    return torch_static_dist, tf_static_dist


@pytest.fixture
def set_base_module() -> Tuple[TorchBayesianModule, TFBayesianModule]:
    """
    This function initializes two instances of Bayesian neural network modules: one for PyTorch and one for TensorFlow.

    Returns:
        Tuple[TorchBayesianModule, TFBayesianModule]:
            A tuple containing two initialized Bayesian neural network modules:
            - torch_module: An instance of TorchBayesianModule.
            - tf_module: An instance of TFBayesianModule.
    """

    # Initialize the PyTorch module
    torch_module = TorchTestModule()

    # Initialize the Tensorflow module
    tf_module = TFTestModule()

    return torch_module, tf_module
