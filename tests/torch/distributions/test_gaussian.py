# 3pp
import pytest
import torch

# own modules
from illia.torch.distributions import GaussianDistribution


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
        shape: shape of the distribution.
        mu_prior: mu for the prior distribution.
        std_prior: std for the prior distribution.
        mu_init: init value for mu. This tensor will be initialized
            with a normal distribution with std 0.1 and the mean is
            the parameter specified here.
        rho_init: init value for rho. This tensor will be initialized
            with a normal distribution with std 0.1 and the mean is
            the parameter specified here.

    Returns:
        None.
    """

    # define distribution
    distribution: GaussianDistribution = GaussianDistribution(shape)

    # check mu prior type
    assert isinstance(distribution.mu_prior, torch.Tensor), (
        f"Incorrect type of mu prior, expected {torch.Tensor}, got "
        f"{type(distribution.mu_prior)}"
    )

    # check std prior type
    assert isinstance(distribution.std_prior, torch.Tensor), (
        f"Incorrect type of std prior, expected {torch.Tensor}, got "
        f"{type(distribution.std_prior)}"
    )

    # check mu type
    assert isinstance(distribution.mu, torch.Tensor), (
        f"Incorrect type of mu, expected {torch.Tensor}, got "
        f"{type(distribution.mu)}"
    )

    # check rho type
    assert isinstance(distribution.rho, torch.Tensor), (
        f"Incorrect type of rho, expected {torch.Tensor}, got "
        f"{type(distribution.rho)}"
    )

    # check number of parameters
    num_parameters: int = len(list(distribution.parameters()))
    assert (
        num_parameters == 2
    ), f"Incorrect number of parameters, expected 2, got {num_parameters}"

    return None


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
        shape: shape of the distribution.

    Returns:
        None.
    """

    # define distribution
    distribution: GaussianDistribution = GaussianDistribution(shape)

    # sample
    sample: torch.Tensor = distribution.sample()

    # check type of sampled tensor
    assert isinstance(sample, torch.Tensor), (
        f"Incorrect type of sample, expected {torch.Tensor}, got " f"{type(sample)}"
    )

    # check shape
    assert (
        sample.shape == shape
    ), f"Incorrect shape, expected {shape}, got {sample.shape}"

    # execute backward pass
    sample.sum().backward()

    # check mu gradients
    assert (
        distribution.mu.grad is not None
    ), "Incorrect backward, mu gradients still None after executing the backward pass"

    # check shape of mu gradients
    assert (
        distribution.mu.grad.shape == shape
    ), f"Incorrect mu grads shape, expected {shape}, got {distribution.mu.grad.shape}"

    # check rho gradients
    assert distribution.rho.grad is not None, (
        "Incorrect backward, rho gradients still None after executing the backward "
        "pass"
    )

    # check shape of rho gradients
    assert distribution.rho.grad.shape == shape, (
        f"Incorrect rho grads shape, expected {shape}, got "
        f"{distribution.rho.grad.shape}"
    )

    return None


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
        shape: shape of the distribution.

    Returns:
        None.
    """

    # define distribution
    distribution: GaussianDistribution = GaussianDistribution(shape)

    # iter over possible x values
    for x in [None, distribution.sample()]:
        # sample
        log_prob: torch.Tensor = distribution.log_prob(x)

        # check type of sampled tensor
        assert isinstance(log_prob, torch.Tensor), (
            f"Incorrect type of log prob, expected {torch.Tensor}, got "
            f"{type(log_prob)}, when input x is {type(x)}"
        )

        # check shape
        assert log_prob.shape == (), (
            f"Incorrect shape of log prob, expected (), got "
            f"{log_prob.shape}, when input x is {type(x)}"
        )

        # execute backward
        log_prob.backward()

        # check mu gradients
        assert distribution.mu.grad is not None, (
            f"Incorrect backward, mu gradients still None after executing the backward "
            f"pass, when input x is {type(x)}"
        )

        # check shape of mu gradients
        assert distribution.mu.grad.shape == shape, (
            f"Incorrect mu grads shape, expected {shape}, got "
            f"{distribution.mu.grad.shape}, when input x is {type(x)}"
        )

        # check rho gradients
        assert distribution.rho.grad is not None, (
            f"Incorrect backward, rho gradients still None after executing the "
            f"backward pass, when input x is {type(x)}"
        )

        # check shape of rho gradients
        assert distribution.rho.grad.shape == shape, (
            f"Incorrect rho grads shape, expected {shape}, got "
            f"{distribution.rho.grad.shape}, when input x is {type(x)}"
        )

    return None


@pytest.mark.parametrize(
    "shape, mu_prior, std_prior, mu_init, rho_init",
    [((32, 30, 20), 0.0, 0.1, 0.0, -0.7), ((64, 3, 32, 32), 0.1, 0.3, 0.1, -0.3)],
)
def test_gaussian_script_jit(
    shape: tuple[int, ...],
    mu_prior: float,
    std_prior: float,
    mu_init: float,
    rho_init: float,
) -> None:
    """
    This functions test the scripting wit torchscript of the
    GaussianDistribution.

    Args:
        shape: shape of the distribution.
    """

    # define distribution
    distribution = torch.jit.script(GaussianDistribution(shape))

    # compute sample
    sample = distribution.sample()

    # check type of sampled tensor
    assert isinstance(sample, torch.Tensor), (
        f"Incorrect type of sample, expected {torch.Tensor}, got " f"{type(sample)}"
    )

    # check shape
    assert (
        sample.shape == shape
    ), f"Incorrect shape, expected {shape}, got {sample.shape}"

    # iter over possible x values
    for x in [None, sample]:
        # sample
        log_prob: torch.Tensor = distribution.log_prob(x)

        # check type of sampled tensor
        assert isinstance(log_prob, torch.Tensor), (
            f"Incorrect type of log prob, expected {torch.Tensor}, got "
            f"{type(log_prob)}, when input x is {type(x)}"
        )

        # check shape
        assert log_prob.shape == (), (
            f"Incorrect shape of log prob, expected (), got "
            f"{log_prob.shape}, when input x is {type(x)}"
        )

    return None
