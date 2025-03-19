import pytest
import jax

from illia.jax.distributions.gaussian import GaussianDistribution


@pytest.mark.parametrize("shape, seed", [((20, 10), 42), ((64, 3, 32, 32), 0)])
def test_gaussian_sample(shape: tuple[int, ...], seed: int) -> None:
    """
    This function is the test for sample method in the jax gaussian
    distribution.

    Returns:
        None.
    """

    # get sample
    distribution: GaussianDistribution = GaussianDistribution(shape, seed)
    sample: jax.Array = distribution.sample()

    # check shape
    assert (
        sample.shape == shape
    ), f"Incorrect shape of array, expected {shape} and got {sample.shape}"

    # check type of object
    assert isinstance(sample, jax.Array), "sample is not a jax array"

    return None


@pytest.mark.parametrize("shape, seed", [((20, 10), 42), ((64, 3, 32, 32), 0)])
def test_gaussian_log_probs(shape: tuple[int, ...], seed: int) -> None:
    """
    This function is the test for log probs method in the jax gaussian
    distribution.

    Returns:
        None.
    """

    # get los probs with no input
    distribution: GaussianDistribution = GaussianDistribution(shape, seed)
    log_probs: jax.Array = distribution.log_prob()

    # check type of object
    assert isinstance(
        log_probs, jax.Array
    ), "log probs is not a jax array when input is None"

    # check shape
    assert (
        log_probs.shape == ()
    ), f"Incorrect shape of array, expected {shape} and got {log_probs.shape}"

    # get los probs with no input
    input_key = jax.random.key(seed)
    inputs: jax.Array = jax.random.normal(input_key, shape)
    log_probs = distribution.log_prob(inputs)

    # check type of object
    assert isinstance(
        log_probs, jax.Array
    ), "log probs is not a jax array when input is not None"

    # check shape
    assert (
        log_probs.shape == ()
    ), f"Incorrect shape of array, expected {shape} and got {log_probs.shape}"

    return None
