"""
This module contains the code to test the GaussianDistribution.
"""

# Standard libraries
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "torch"


# 3pps
import pytest
import torch
from torch.jit import RecursiveScriptModule

# Own modules
from illia.distributions import GaussianDistribution


@pytest.mark.parametrize(
    "shape, mu_prior, std_prior, mu_init, rho_init",
    [((32, 30, 20), 0.0, 0.1, 0.0, -0.7), ((64, 3, 32, 32), 0.1, 0.3, 0.1, -0.3)],
)
class TestGaussianDistribution:
    """
    This class implements the tests for the GaussianDistribution class.
    """

    @pytest.mark.order(1)
    def test_init(
        self,
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
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape,
            mu_prior,
            std_prior,
            mu_init,
            rho_init,
        )

        # Check mu prior type
        assert isinstance(distribution.mu_prior, torch.Tensor), (
            f"Incorrect type of mu prior, expected {torch.Tensor}, got "
            f"{type(distribution.mu_prior)}"
        )

        # Check std prior type
        assert isinstance(distribution.std_prior, torch.Tensor), (
            f"Incorrect type of std prior, expected {torch.Tensor}, got "
            f"{type(distribution.std_prior)}"
        )

        # Check mu type
        assert isinstance(distribution.mu, torch.Tensor), (
            f"Incorrect type of mu, expected {torch.Tensor}, got "
            f"{type(distribution.mu)}"
        )

        # Check rho type
        assert isinstance(distribution.rho, torch.Tensor), (
            f"Incorrect type of rho, expected {torch.Tensor}, got "
            f"{type(distribution.rho)}"
        )

        # Check number of parameters
        num_parameters: int = len(list(distribution.parameters()))
        assert (
            num_parameters == 2
        ), f"Incorrect number of parameters, expected 2, got {num_parameters}"

    @pytest.mark.order(2)
    def test_sample(
        self,
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
            mu_prior: mu for the prior distribution.
            std_prior: std for the prior distribution.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
            rho_init: init value for rho. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape,
            mu_prior,
            std_prior,
            mu_init,
            rho_init,
        )

        # Sample
        sample: torch.Tensor = distribution.sample()

        # Check type of sampled tensor
        assert isinstance(sample, torch.Tensor), (
            f"Incorrect type of sample, expected {torch.Tensor}, got " f"{type(sample)}"
        )

        # Check shape
        assert (
            sample.shape == shape
        ), f"Incorrect shape, expected {shape}, got {sample.shape}"

        # Execute backward pass
        sample.sum().backward()

        # Check mu gradients
        assert distribution.mu.grad is not None, (
            "Incorrect backward, mu gradients still None after executing the backward "
            "pass"
        )

        # Check shape of mu gradients
        assert distribution.mu.grad.shape == shape, (
            f"Incorrect mu grads shape, expected {shape}, got "
            f"{distribution.mu.grad.shape}"
        )

        # Check rho gradients
        assert distribution.rho.grad is not None, (
            "Incorrect backward, rho gradients still None after executing the "
            "backward pass"
        )

        # Check shape of rho gradients
        assert distribution.rho.grad.shape == shape, (
            f"Incorrect rho grads shape, expected {shape}, got "
            f"{distribution.rho.grad.shape}"
        )

    @pytest.mark.order(3)
    def test_log_prob(
        self,
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
            mu_prior: mu for the prior distribution.
            std_prior: std for the prior distribution.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
            rho_init: init value for rho. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape,
            mu_prior,
            std_prior,
            mu_init,
            rho_init,
        )

        # Iter over possible x values
        for x in [None, distribution.sample()]:
            # Sample
            log_prob: torch.Tensor = distribution.log_prob(x)

            # Check type of sampled tensor
            assert isinstance(log_prob, torch.Tensor), (
                f"Incorrect type of log prob, expected {torch.Tensor}, got "
                f"{type(log_prob)}, when input x is {type(x)}"
            )

            # Check shape
            assert log_prob.shape == (), (
                f"Incorrect shape of log prob, expected (), got "
                f"{log_prob.shape}, when input x is {type(x)}"
            )

            # Execute backward
            log_prob.backward()

            # Check mu gradients
            assert distribution.mu.grad is not None, (
                f"Incorrect backward, mu gradients still None after executing the "
                f"backward pass, when input x is {type(x)}"
            )

            # Check shape of mu gradients
            assert distribution.mu.grad.shape == shape, (
                f"Incorrect mu grads shape, expected {shape}, got "
                f"{distribution.mu.grad.shape}, when input x is {type(x)}"
            )

            # Check rho gradients
            assert distribution.rho.grad is not None, (
                f"Incorrect backward, rho gradients still None after executing the "
                f"backward pass, when input x is {type(x)}"
            )

            # Check shape of rho gradients
            assert distribution.rho.grad.shape == shape, (
                f"Incorrect rho grads shape, expected {shape}, got "
                f"{distribution.rho.grad.shape}, when input x is {type(x)}"
            )

    @pytest.mark.order(4)
    def test_num_params(
        self,
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
            mu_prior: mu for the prior distribution.
            std_prior: std for the prior distribution.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
            rho_init: init value for rho. This tensor will be
                initialized with a normal distribution with std 0.1
                and the mean is the parameter specified here.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape,
            mu_prior,
            std_prior,
            mu_init,
            rho_init,
        )

        # Compute num params
        num_params: int = distribution.num_params()
        num_params_correct: int = len(distribution.mu.view(-1))

        # Check number of params
        assert num_params == num_params_correct, (
            f"Incorrect number of parameters, expected {num_params_correct} and got "
            f"{num_params}"
        )

    @pytest.mark.order(5)
    def test_change_device(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
    ) -> None:
        """
        This test checks the change of device of the
        GaussianDistribution

        Args:
            shape: shape of the distribution.
            mu_prior: mu for the prior distribution.
            std_prior: std for the prior distribution.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
            rho_init: init value for rho. This tensor will be
                initialized with a normal distribution with std 0.1
                and the mean is the parameter specified here.
        """

        # Define two devices
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape,
            mu_prior,
            std_prior,
            mu_init,
            rho_init,
        )

        # Change device and sample
        distribution = distribution.to(device)
        sample = distribution.sample()

        # Check device of sample
        assert sample.device == device, "Incorrect outputs device when device changed"

    @pytest.mark.order(6)
    def test_jit(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
    ) -> None:
        """
        This test checks the change of device of the GaussianDistribution

        Args:
            shape: shape of the distribution.
            mu_prior: mu for the prior distribution.
            std_prior: std for the prior distribution.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
            rho_init: init value for rho. This tensor will be
                initialized with a normal distribution with std 0.1
                and the mean is the parameter specified here.
        """

        # Define distribution and script
        distribution: RecursiveScriptModule = torch.jit.script(
            GaussianDistribution(
                shape,
                mu_prior,
                std_prior,
                mu_init,
                rho_init,
            )
        )

        # Change device and sample
        sample: torch.Tensor = distribution.sample()
        log_prob: torch.Tensor = distribution.log_prob()
        num_params: int = distribution.num_params()

        # Check shape and type
        assert sample.shape == shape, "Incorrect sample shape when scripting"
        assert log_prob.shape == (), "Incorrect log_prob shape when scripting"
        assert isinstance(
            num_params, int
        ), f"Incorrect type, expected {int} and got {type(num_params)}"
