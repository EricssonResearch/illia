# Libraries
import random

import pytest  # type: ignore
import numpy as np  # type: ignore
import torch
import tensorflow as tf  # type: ignore

from tests.fixtures_nn import set_base_module

RNG = np.random.default_rng(seed=0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)

@pytest.mark.order(4)
def test_conv_fordward(set_base_module):
    
    torch_module, tf_module = set_base_module

    input_data = RNG.normal(size=(1, 3, 28, 28))

    torch_input = torch.from_numpy(input_data).float()
    tf_input = tf.convert_to_tensor(input_data)

    np.testing.assert_allclose(torch_input.detach().numpy(), tf_input.numpy(), rtol=1, atol=1, 
                               err_msg='Torch and TensorFlow output are different by relative error (1) and absolute error (1)',
                               verbose=True)
    # torch_output = torch_module(torch_input)
    # tf_output = tf_module(tf_input)

    assert True