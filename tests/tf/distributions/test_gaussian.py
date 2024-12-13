# 3pp
import pytest
import tensorflow as tf

# own modules
from illia.tf.distributions import GaussianDistribution


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

    # define model
    model: GaussianDistribution = GaussianDistribution(shape)

    # check mu prior type
    assert isinstance(model.mu_prior, tf.Tensor), (
        f"Incorrect type of mu prior, expected {tf.Tensor}, got "
        f"{type(model.mu_prior)}"
    )

    # check std prior type
    assert isinstance(model.std_prior, tf.Tensor), (
        f"Incorrect type of std prior, expected {tf.Tensor}, got "
        f"{type(model.std_prior)}"
    )

    # check mu type
    assert isinstance(model.mu, tf.keras.Variable), (
        f"Incorrect type of mu, expected {tf.keras.Variable}, got " f"{type(model.mu)}"
    )

    # check rho type
    assert isinstance(model.rho, tf.keras.Variable), (
        f"Incorrect type of rho, expected {tf.keras.Variable}, got "
        f"{type(model.rho)}"
    )

    # check number of parameters
    num_parameters: int = len(model.trainable_variables)
    assert (
        num_parameters == 2
    ), f"Incorrect number of parameters, expected 2, got {num_parameters}"

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

    # define model
    model: GaussianDistribution = GaussianDistribution(shape)

    # execute backward pass
    with tf.GradientTape() as tape:
        sample: tf.Tensor = model.sample()
    gradients = tape.gradient(sample, model.trainable_variables)

    # check type of sampled tensor
    assert isinstance(sample, tf.Tensor), (
        f"Incorrect type of sample, expected {tf.Tensor}, got " f"{type(sample)}"
    )

    # check shape
    assert (
        sample.shape == shape
    ), f"Incorrect shape, expected {shape}, got {sample.shape}"

    # check number of gradients
    num_gradients: int = len(gradients)
    assert (
        num_gradients == 2
    ), f"Incorrect number of gradients, expected 2, got {num_gradients}"

    # check gradients shape
    for i, gradient in enumerate(gradients):
        # check shape of gradients
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

    # define model
    model: GaussianDistribution = GaussianDistribution(shape)

    # iter over possible x values
    for x in [None, model.sample()]:
        # execute forward & backward pass
        with tf.GradientTape() as tape:
            log_prob: tf.Tensor = model.log_prob(x)
        gradients = tape.gradient(log_prob, model.trainable_variables)

        # check type of sampled tensor
        assert isinstance(log_prob, tf.Tensor), (
            f"Incorrect type of log prob, expected {tf.Tensor}, got "
            f"{type(log_prob)}, when input x is {type(x)}"
        )

        # check shape
        assert log_prob.shape == (), (
            f"Incorrect shape of log prob, expected (), got "
            f"{log_prob.shape}, when input x is {type(x)}"
        )

        # check number of gradients
        num_gradients: int = len(gradients)
        assert (
            num_gradients == 2
        ), f"Incorrect number of gradients, expected 2, got {num_gradients}"

        # check gradients shape
        for i, gradient in enumerate(gradients):
            # check shape of gradients
            assert gradient.shape == model.trainable_variables[i].shape, (
                f"Incorrect mu grads shape, expected {shape}, got "
                f"{model.trainable_variables[i].shape}"
            )

    return None
