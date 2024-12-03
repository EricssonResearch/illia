# Libraries
import random
from typing import Dict, Tuple

import pytest
import numpy as np
import torch
import tensorflow as tf

# Specific libraries for each backend
from illia.distributions.static.tf.gaussian import GaussianDistribution as TFStaticGaussianDistribution
from illia.distributions.static.torch.gaussian import GaussianDistribution as TorchStaticGaussianDistribution

from illia.distributions.dynamic.tf.gaussian import GaussianDistribution as TFDynamicGaussianDistribution
from illia.distributions.dynamic.torch.gaussian import GaussianDistribution as TorchDynamicGaussianDistribution

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


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
    torch_dynamic_dist = TorchDynamicGaussianDistribution(
        shape=shape, mu_init=mu_init, rho_init=rho_init
    )
    tf_dynamic_dist = TFDynamicGaussianDistribution(
        shape=shape, mu_init=mu_init, rho_init=rho_init
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
    torch_static_dist = TorchStaticGaussianDistribution(
        mu=mu_prior, std=std_prior
    )
    tf_static_dist = TFStaticGaussianDistribution(
        mu=mu_prior, std=std_prior
    )

    return torch_static_dist, tf_static_dist
