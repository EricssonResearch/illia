# Libraries
import random

import pytest
import numpy as np
import torch
import tensorflow as tf
from keras import layers

from . import TorchBayesianModule, TFBayesianModule

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


@pytest.fixture
def set_base_module() -> tuple:
    """
    Creates test instances of Bayesian modules for both PyTorch and
    TensorFlow frameworks. Each module includes a linear layer and
    methods `forward` or `call` for passing data through the network,
    as well as `kl_cost` for calculating the KL divergence cost.
    Returns a tuple of PyTorch and TensorFlow module instances.
    """

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
            self.linear = layers.Dense(5)

        def call(self, x):
            return self.linear(x)

        def kl_cost(self):
            return tf.constant(1.0), 1

    torch_module = TorchTestModule()
    tf_module = TFTestModule()

    return torch_module, tf_module
