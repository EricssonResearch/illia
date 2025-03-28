"""
This module contains the code to test the GaussianDistribution.
"""

# 3pps
import tensorflow as tf
import pytest
from keras.src.backend.tensorflow.core import Variable as BackendVariable

# Own modules
from illia.tf.distributions import GaussianDistribution


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
        This test is to check the constructor of GaussianDistribution
        class.

        Args:
            shape: shape of the distribution.
            mu_prior: mu for the prior distribution.
            std_prior: std for the prior distribution.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here.
            rho_init: init value for rho. This tensor will be
                initialized with a normal distribution with std 0.1 and
                the mean is the parameter specified here.
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
        assert isinstance(distribution.mu_prior, BackendVariable), (
            f"Incorrect type of mu prior, expected {BackendVariable}, got "
            f"{type(distribution.mu_prior)}"
        )

        # Check std prior type
        assert isinstance(distribution.std_prior, BackendVariable), (
            f"Incorrect type of std prior, expected {BackendVariable}, got "
            f"{type(distribution.std_prior)}"
        )

        # Check mu type
        assert isinstance(distribution.mu, BackendVariable), (
            f"Incorrect type of mu, expected {BackendVariable}, got "
            f"{type(distribution.mu)}"
        )

        # Check rho type
        assert isinstance(distribution.rho, BackendVariable), (
            f"Incorrect type of rho, expected {BackendVariable}, got "
            f"{type(distribution.rho)}"
        )

        # Check number of parameters
        num_parameters: int = len(distribution.trainable_variables)
        assert (
            num_parameters == 2
        ), f"Incorrect number of parameters, expected 2, got {num_parameters}"

        # Check the shape of the initialized tensors
        assert (
            distribution.mu_prior.shape == ()
        ), f"Incorrect shape of mu_prior got {distribution.mu_prior.shape}"
        assert (
            distribution.std_prior.shape == ()
        ), f"Incorrect shape of std_prior got {distribution.mu_prior.shape}"
        assert (
            distribution.mu.shape == shape
        ), f"Incorrect shape of mu got {distribution.mu.shape}"
        assert (
            distribution.rho.shape == shape
        ), f"Incorrect shape of rho got {distribution.rho.shape}"

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
        with tf.GradientTape() as tape:
            sample: tf.Tensor = distribution.sample()
        gradients = tape.gradient(sample, distribution.trainable_variables)

        # Check type of sampled tensor
        assert isinstance(
            sample, tf.Tensor
        ), f"Incorrect type of sample, expected {tf.Tensor}, got {type(sample)}"

        # Check shape
        assert (
            sample.shape == shape
        ), f"Incorrect shape, expected {shape}, got {sample.shape}"

        # Check gradients shape
        for i, gradient in enumerate(gradients):
            assert gradient.shape == distribution.trainable_variables[i].shape, (
                f"Incorrect mu grads shape, expected {shape}, got "
                f"{distribution.trainable_variables[i].shape}"
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
            with tf.GradientTape() as tape:
                log_prob: tf.Tensor = distribution.log_prob(x)
            gradients = tape.gradient(log_prob, distribution.trainable_variables)

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

            # Check gradients shape
            for i, gradient in enumerate(gradients):
                assert gradient.shape == distribution.trainable_variables[i].shape, (
                    f"Incorrect mu grads shape, expected {shape}, got "
                    f"{distribution.trainable_variables[i].shape}"
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

        # Compute num params
        num_params: int = distribution.num_params
        num_params_correct: int = int(tf.reshape(distribution.mu, [-1]).shape[0])

        # Check number of params
        assert num_params == num_params_correct, (
            f"Incorrect number of parameters, expected {num_params_correct} and got "
            f"{num_params}"
        )
