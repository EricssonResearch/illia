# Libraries
import random
from typing import Dict, Tuple

import pytest
import numpy as np
import torch
import tensorflow as tf

from illia.torch.nn.base import BayesianModule as TorchBayesianModule
from illia.tf.nn.base import BayesianModule as TFBayesianModule

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


@pytest.fixture
def set_base_module() -> Tuple:

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
            self.linear = tf.keras.layers.Dense(5)

        def call(self, x):
            return self.linear(x)

        def kl_cost(self):
            return tf.constant(1.0), 1

    torch_module = TorchTestModule()
    tf_module = TFTestModule()

    return torch_module, tf_module
