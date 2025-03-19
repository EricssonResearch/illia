import random
from typing import Dict, Tuple

import pytest
import torch
import numpy as np
import tensorflow as tf

from . import TFGaussianDistribution, TorchGaussianDistribution

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


@pytest.fixture
def set_parameters() -> Dict:
    """
    Returns a dictionary of common parameters used for initializing
    Gaussian distributions. The parameters include the shape, prior
    mean and standard deviation, initial mean, and initial rho value.
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
def set_distributions(set_parameters) -> Tuple:
    """
    Initializes dynamic Gaussian distributions using the parameters
    provided by `set_parameters`. Returns a tuple containing both
    PyTorch and TensorFlow dynamic Gaussian distribution instances.
    """

    # Access to the variables
    shape = set_parameters["shape"]
    mu_init = set_parameters["mu_init"]
    rho_init = set_parameters["rho_init"]

    # Initialize dynamic distributions
    torch_dist = TorchGaussianDistribution(
        shape=shape, mu_init=mu_init, rho_init=rho_init
    )
    tf_dist = TFGaussianDistribution(shape=shape, mu_init=mu_init, rho_init=rho_init)

    return torch_dist, tf_dist
