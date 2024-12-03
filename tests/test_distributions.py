# Libraries
import random

import pytest
import numpy as np
import torch
import tensorflow as tf

from tests.fixtures_distributions import (
    set_parameters,
    set_dynamic_distributions,
    set_static_distributions,
)
from tests.utils.utils import compare_tensors

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


@pytest.mark.order(1)
@pytest.mark.parametrize("n_samples", [10000])
def test_dynamic_sampling(n_samples, set_dynamic_distributions) -> None:
    """
    This function tests the dynamic sampling of distributions.

    Args:
        n_samples (int): The number of samples to generate for each distribution.
        set_dynamic_distributions (tuple): A tuple containing the dynamic distributions.
    """

    # Obtain the dynamic distributions
    torch_dist, tf_dist = set_dynamic_distributions

    # Sampling the data
    torch_samples = np.array(
        [torch_dist.sample().detach().cpu().numpy() for _ in range(n_samples)]
    )
    tf_samples = np.array([tf_dist.sample().numpy() for _ in range(n_samples)])

    # Compare means
    torch_mean = np.mean(torch_samples, axis=0)
    tf_mean = np.mean(tf_samples, axis=0)
    assert (
        compare_tensors(torch_mean, tf_mean, name="Means") == True
    ), "The mean of the tensors of the dynamic distributions are not similar."

    # Compare standard deviations
    torch_std = np.std(torch_samples, axis=0)
    tf_std = np.std(tf_samples, axis=0)
    assert (
        compare_tensors(torch_std, tf_std, name="Standard deviations") == True
    ), "The standard deviation of the tensors of the dynamic distributions are not similar."


@pytest.mark.order(2)
@pytest.mark.parametrize("rtol, atol", [(1e-1, 1e-1)])
def test_dynamic_log_probs(
    rtol, atol, set_parameters, set_dynamic_distributions
) -> None:
    """
    This function tests the log probabilities of dynamic distributions.

    Args:
        rtol (float): The relative tolerance for comparing the log probabilities.
        atol (float): The absolute tolerance for comparing the log probabilities.
        set_parameters (dict): A dictionary containing parameters required for the test.
        set_dynamic_distributions (tuple): A tuple containing the dynamic distributions.
    """

    # Access to the variables
    shape = set_parameters["shape"]

    # Obtain the dynamic distributions
    torch_dist, tf_dist = set_dynamic_distributions

    # Sampling the data, computing log probability and compare it
    x = np.random.randn(*shape).astype(np.float32)
    torch_log_prob = (
        torch_dist.log_prob(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    )
    tf_log_prob = tf_dist.log_prob(tf.constant(x, dtype=tf.float32)).numpy()
    assert (
        compare_tensors(
            torch_log_prob, tf_log_prob, rtol=rtol, atol=atol, name="Log probabilities"
        )
        == True
    ), "The log probability of the tensors of the dynamic distributions are not similar."


@pytest.mark.order(3)
def test_dynamic_num_params(set_dynamic_distributions) -> None:
    """
    This function tests the number of parameters of dynamic distributions.

    Args:
        set_dynamic_distributions (tuple): A tuple containing the dynamic distributions.
    """

    # Obtain the dynamic distributions
    torch_dist, tf_dist = set_dynamic_distributions

    # Compare the number of parameters
    assert (
        torch_dist.num_params == tf_dist.num_params
    ), "The number of parameters of the dynamic distributions are not similar."


@pytest.mark.order(4)
@pytest.mark.parametrize("n_samples, rtol, atol", [(10000, 1e-1, 1e-1)])
def test_static_log_probs(n_samples, rtol, atol, set_static_distributions) -> None:
    """
    This function tests the log probabilities of static distributions.

    Args:
        n_samples (int): The number of samples to generate for each distribution.
        rtol (float): The relative tolerance for comparing the log probabilities.
        atol (float): The absolute tolerance for comparing the log probabilities.
        set_static_distributions (tuple): A tuple containing the static distributions.
    """

    # Obtain the static distributions
    torch_dist, tf_dist = set_static_distributions

    # Sampling the data, computing log probability and compare it
    x = np.random.randn(n_samples).astype(np.float32)
    torch_log_prob = (
        torch_dist.log_prob(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    )
    tf_log_prob = tf_dist.log_prob(tf.constant(x, dtype=tf.float32)).numpy()
    assert (
        compare_tensors(
            torch_log_prob, tf_log_prob, rtol=rtol, atol=atol, name="Log probabilities"
        )
        == True
    ), "The log probability of the tensors of the static distributions are not similar."
