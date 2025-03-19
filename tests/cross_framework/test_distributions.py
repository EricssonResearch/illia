import random

import pytest
import numpy as np
import torch
import tensorflow as tf

from .fixtures_distributions import set_parameters, set_distributions

from .utils import compare_tensors

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


@pytest.mark.order(1)
@pytest.mark.parametrize("n_samples", [10000])
def test_dynamic_sampling(n_samples, set_distributions) -> None:
    """
    This function samples data from both PyTorch and TensorFlow dynamic
    distributions and compares their means and standard deviations to
    ensure similarity.

    Args:
        n_samples: Number of samples to draw.
        set_distributions: Fixture to provide dynamic
            distributions.

    Raises:
        AssertionError: If means or standard deviations of the sampled
        data do not match between the dynamic distributions.
    """

    # Obtain the dynamic distributions
    torch_dist, tf_dist = set_distributions

    # Sampling the data
    torch_samples = np.array(
        [torch_dist.sample().detach().cpu().numpy() for _ in range(n_samples)]
    )
    tf_samples = np.array([tf_dist.sample().numpy() for _ in range(n_samples)])

    # Compare means
    torch_mean = np.mean(torch_samples, axis=0)
    tf_mean = np.mean(tf_samples, axis=0)
    assert compare_tensors(
        torch_mean, tf_mean, name="Means"
    ), "The mean of the tensors of the dynamic distributions are not similar."

    # Compare standard deviations
    torch_std = np.std(torch_samples, axis=0)
    tf_std = np.std(tf_samples, axis=0)
    assert compare_tensors(
        torch_std, tf_std, name="Standard deviations"
    ), "The std of the tensors of the dynamic distributions are not similar."


@pytest.mark.order(2)
@pytest.mark.parametrize("rtol, atol", [(1e-1, 1e-1)])
def test_dynamic_log_probs(rtol, atol, set_parameters, set_distributions) -> None:
    """
    This function computes the log probabilities of samples from both
    PyTorch and TensorFlow dynamic distributions and compares them for
    similarity.

    Args:
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        set_parameters: Fixture to provide parameters including shape.
        set_distributions: Fixture to provide dynamic
            distributions.

    Raises:
        AssertionError: If log probabilities do not match between the
        dynamic distributions.
    """

    # Access to the variables
    shape = set_parameters["shape"]

    # Obtain the dynamic distributions
    torch_dist, tf_dist = set_distributions

    # Sampling the data, computing log probability and compare it
    x = np.random.randn(*shape).astype(np.float32)
    torch_log_prob = (
        torch_dist.log_prob(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    )
    tf_log_prob = tf_dist.log_prob(tf.constant(x, dtype=tf.float32)).numpy()
    assert compare_tensors(
        torch_log_prob, tf_log_prob, rtol=rtol, atol=atol, name="Log probabilities"
    ), "The log prob. of the tensors of the dynamic distributions are not similar."


@pytest.mark.order(3)
def test_dynamic_num_params(set_distributions) -> None:
    """
    This function compares the number of parameters between PyTorch and
    TensorFlow dynamic distributions to ensure they are similar.

    Args:
        set_distributions: Fixture to provide dynamic
            distributions.

    Raises:
        AssertionError: If the number of parameters do not match between
        the dynamic distributions.
    """

    # Obtain the dynamic distributions
    torch_dist, tf_dist = set_distributions

    # Compare the number of parameters
    assert (
        torch_dist.num_params == tf_dist.num_params
    ), "The number of parameters of the dynamic distributions are not similar."


@pytest.mark.order(4)
@pytest.mark.parametrize("n_samples, rtol, atol", [(10000, 1e-1, 1e-1)])
def test_static_log_probs(n_samples, rtol, atol, set_static_distributions) -> None:
    """
    This function computes the log probabilities of samples from both
    PyTorch and TensorFlow static distributions and compares them for
    similarity.

    Args:
        n_samples: Number of samples to draw.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        set_static_distributions: Fixture to provide static
            distributions.

    Raises:
        AssertionError: If log probabilities do not match between the
        static distributions.
    """

    # Obtain the static distributions
    torch_dist, tf_dist = set_static_distributions

    # Sampling the data, computing log probability and compare it
    x = np.random.randn(n_samples).astype(np.float32)
    torch_log_prob = (
        torch_dist.log_prob(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    )
    tf_log_prob = tf_dist.log_prob(tf.constant(x, dtype=tf.float32)).numpy()
    assert compare_tensors(
        torch_log_prob, tf_log_prob, rtol=rtol, atol=atol, name="Log probabilities"
    ), "The log prob. of the tensors of the static distributions are not similar."
