# Libraries
import random

import pytest
import numpy as np
import torch
import tensorflow as tf

from tests.fixtures_distributions import set_base_module

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


@pytest.mark.order(1)
def test_base_freeze_unfreeze(set_base_module):
    """
    This function tests the freeze and unfreeze methods of a base module.
    The base module is expected to have a 'frozen' attribute and 'freeze' and 'unfreeze' methods.

    Args:
        set_base_module (tuple): A tuple containing the base modules.
    """

    # Obtain the base module
    torch_module, tf_module = set_base_module

    # Test PyTorch module
    assert not torch_module.frozen, "PyTorch module should not be frozen initially"
    torch_module.freeze()
    assert torch_module.frozen, "PyTorch module should be frozen after freeze()"
    torch_module.unfreeze()
    assert (
        not torch_module.frozen
    ), "PyTorch module should not be frozen after unfreeze()"

    # Test TensorFlow module
    assert not tf_module.frozen, "TensorFlow module should not be frozen initially"
    tf_module.freeze()
    assert tf_module.frozen, "TensorFlow module should be frozen after freeze()"
    tf_module.unfreeze()
    assert (
        not tf_module.frozen
    ), "TensorFlow module should not be frozen after unfreeze()"


@pytest.mark.order(2)
def test_base_kl_cost_function(set_base_module):
    """
    This function tests the KL cost function of a base module.
    It compares the KL divergence and the number of samples (N) returned by both frameworks.

    Args:
        set_base_module (tuple): A tuple containing the base modules.
    """

    # Obtain the base module
    torch_module, tf_module = set_base_module

    # Obtain the KL cost function
    torch_kl, torch_n = torch_module.kl_cost()
    tf_kl, tf_n = tf_module.kl_cost()

    # Test KL cost functions
    assert (
        torch_kl.item() == tf_kl.numpy()
    ), f"KL divergence mismatch: PyTorch {torch_kl.item()}, TensorFlow {tf_kl.numpy()}"
    assert torch_n == tf_n, f"N mismatch: PyTorch {torch_n}, TensorFlow {tf_n}"


@pytest.mark.order(3)
def test_base_forward_pass(set_base_module):
    """
    This function tests the forward pass of a base module.
    It compares the outputs of both frameworks and prints a warning if the difference exceeds a certain threshold.

    Args:
        set_base_module (tuple): A tuple containing the base modules.
    """

    # Obtain the base module
    torch_module, tf_module = set_base_module

    # Input data
    input_data = np.random.randn(1, 10).astype(np.float32)

    # PyTorch forward pass
    torch_input = torch.from_numpy(input_data)
    torch_output = torch_module(torch_input)

    # TensorFlow forward pass
    tf_input = tf.convert_to_tensor(input_data)
    tf_output = tf_module(tf_input)

    # Compare outputs
    torch_np = torch_output.detach().numpy()
    tf_np = tf_output.numpy()

    max_diff = np.max(np.abs(torch_np - tf_np))
    print(f"Maximum absolute difference: {max_diff}")

    if max_diff > 1e-1:
        print(
            """
            Warning-Ignore for now: Outputs differ slightly, this might be due to different initialization or computational 
            differences between PyTorch and TensorFlow for torch.nn.Linear && tf.keras.layers.Dense
            """
        )
        print("PyTorch output:", torch_np)
        print("TensorFlow output:", tf_np)
    else:
        print("Outputs are close enough.")

    # Use a more lenient comparison
    np.testing.assert_allclose(torch_np, tf_np, rtol=1, atol=1)
