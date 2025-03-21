# 3pp
import pytest
import numpy as np
import tensorflow as tf
from keras.src.backend.tensorflow.core import Variable as BackendVariable

# Own modules
from illia.tf.distributions.gaussian import GaussianDistribution


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "shape, mu_prior, std_prior, mu_init, rho_init",
    [((32, 30, 20), 0.0, 0.1, 0.0, -0.7), ((64, 3, 32, 32), 0.1, 0.3, 0.1, -0.3)],
)
def test_gaussian_init(
    shape: tuple[int, ...],
    mu_prior: float,
    std_prior: float,
    mu_init: float,
    rho_init: float,
) -> None:
    """
    This test is to check the constructor of GaussianDistribution class.

    Args:
        shape: shape of the model.
        mu_prior: mu for the prior model.
        std_prior: std for the prior model.
        mu_init: init value for mu. This tensor will be initialized
            with a normal model with std 0.1 and the mean is
            the parameter specified here.
        rho_init: init value for rho. This tensor will be initialized
            with a normal model with std 0.1 and the mean is
            the parameter specified here.

    Returns:
        None.
    """

    # Define model
    model: GaussianDistribution = GaussianDistribution(
        shape=shape,
        mu_prior=mu_prior,
        std_prior=std_prior,
        mu_init=mu_init,
        rho_init=rho_init,
    )

    # Check mu prior type
    assert isinstance(model.mu_prior, tf.Tensor), (
        f"Incorrect type of mu prior, expected {tf.Tensor}, got "
        f"{type(model.mu_prior)}"
    )

    # Check std prior type
    assert isinstance(model.std_prior, tf.Tensor), (
        f"Incorrect type of std prior, expected {tf.Tensor}, got "
        f"{type(model.std_prior)}"
    )

    # Check mu type
    assert isinstance(
        model.mu, BackendVariable
    ), f"Incorrect type of mu, expected {BackendVariable}, got {type(model.mu)}"

    # Check rho type
    assert isinstance(
        model.rho, BackendVariable
    ), f"Incorrect type of rho, expected {BackendVariable}, got {type(model.rho)}"

    # Check number of parameters
    num_parameters: int = len(model.trainable_variables)
    assert (
        num_parameters == 2
    ), f"Incorrect number of parameters, expected 2, got {num_parameters}"

    # Check the shape of the initialized tensors
    assert model.mu_prior.shape == (), "Incorrect shape of mu_prior"
    assert model.std_prior.shape == (), "Incorrect shape of std_prior"
    assert model.mu.shape == shape, "Incorrect shape of mu"
    assert model.rho.shape == shape, "Incorrect shape of rho"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "shape, mu_prior, std_prior, mu_init, rho_init",
    [((32, 30, 20), 0.0, 0.1, 0.0, -0.7), ((64, 3, 32, 32), 0.1, 0.3, 0.1, -0.3)],
)
def test_gaussian_sample(
    shape: tuple[int, ...],
    mu_prior: float,
    std_prior: float,
    mu_init: float,
    rho_init: float,
) -> None:
    """
    This test checks the sample method of GaussianDistribution.

    Args:
        shape: shape of the model.

    Returns:
        None.
    """

    # Define model
    model: GaussianDistribution = GaussianDistribution(
        shape=shape,
        mu_prior=mu_prior,
        std_prior=std_prior,
        mu_init=mu_init,
        rho_init=rho_init,
    )

    # Execute backward pass
    with tf.GradientTape() as tape:
        sample: tf.Tensor = model.sample()
    gradients = tape.gradient(sample, model.trainable_variables)

    # Check type of sampled tensor
    assert isinstance(sample, tf.Tensor), (
        f"Incorrect type of sample, expected {tf.Tensor}, got " f"{type(sample)}"
    )

    # Check shape
    assert (
        sample.shape == shape
    ), f"Incorrect shape, expected {shape}, got {sample.shape}"

    # Check number of gradients
    num_gradients: int = len(gradients)
    assert (
        num_gradients == 2
    ), f"Incorrect number of gradients, expected 2, got {num_gradients}"

    # Check gradients shape
    for i, gradient in enumerate(gradients):
        # Check shape of gradients
        assert gradient.shape == model.trainable_variables[i].shape, (
            f"Incorrect mu grads shape, expected {shape}, got "
            f"{model.trainable_variables[i].shape}"
        )

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "shape, mu_prior, std_prior, mu_init, rho_init",
    [((32, 30, 20), 0.0, 0.1, 0.0, -0.7), ((64, 3, 32, 32), 0.1, 0.3, 0.1, -0.3)],
)
def test_gaussian_log_prob(
    shape: tuple[int, ...],
    mu_prior: float,
    std_prior: float,
    mu_init: float,
    rho_init: float,
) -> None:
    """
    This test checks the log_prob method of GaussianDistribution.

    Args:
        shape: shape of the model.

    Returns:
        None.
    """

    # Define model
    model: GaussianDistribution = GaussianDistribution(
        shape=shape,
        mu_prior=mu_prior,
        std_prior=std_prior,
        mu_init=mu_init,
        rho_init=rho_init,
    )

    # Iter over possible x values
    for x in [None, model.sample()]:
        # Execute forward & backward pass
        with tf.GradientTape() as tape:
            log_prob: tf.Tensor = model.log_prob(x)
        gradients = tape.gradient(log_prob, model.trainable_variables)

        # Check type of sampled tensor
        assert isinstance(log_prob, tf.Tensor), (
            f"Incorrect type of log prob, expected {tf.Tensor}, got "
            f"{type(log_prob)}, when input x is {type(x)}"
        )

        # Check shape
        assert log_prob.shape == (), (
            f"Incorrect shape of log prob, expected (), got "
            f"{log_prob.shape}, when input x is {type(x)}"
        )

        # Check number of gradients
        num_gradients: int = len(gradients)
        assert (
            num_gradients == 2
        ), f"Incorrect number of gradients, expected 2, got {num_gradients}"

        # Check gradients shape
        for i, gradient in enumerate(gradients):
            # Check shape of gradients
            assert gradient.shape == model.trainable_variables[i].shape, (
                f"Incorrect mu grads shape, expected {shape}, got "
                f"{model.trainable_variables[i].shape}"
            )

    return None
