"""
This module contains the tests for GaussianDistribution.
"""

# Standard libraries
import copy
import os


# Change Illia Backend
os.environ["ILLIA_BACKEND"] = "jax"

# 3pps
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from flax.nnx.rnglib import Rngs

# Own modules
from illia.distributions import GaussianDistribution


@pytest.mark.parametrize(
    "shape, mu_prior, std_prior, mu_init, rho_init, rngs",
    [
        ((20, 30), 0.0, 0.1, 0.0, -7.0, nnx.Rngs(0)),
        ((3, 16, 16), 0.5, 1.2, 0.5, 3.0, nnx.Rngs(1)),
    ],
)
class TestGaussianDistribution:
    @pytest.mark.order(1)
    def test_init(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
        rngs: Rngs,
    ) -> None:
        """
        This method tests the init method.

        Args:
            shape: Shape of the weights.
            mu_prior: Mu value prior.
            std_prior: Std value prior.
            mu_init: Mu init prior.
            rho_init: Rho init prior.
            rngs: _description_

        Returns:
            None.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape, mu_prior, std_prior, mu_init, rho_init, rngs
        )

        # Compute number of parameters
        params = nnx.state(distribution, nnx.Param)
        num_params: int = len(params)

        # Check number of parameters
        assert (
            num_params == 2
        ), f"Incorrect number of parameters, expected 2 and got {num_params}"

        return None

    @pytest.mark.order(2)
    def test_sample(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
        rngs: Rngs,
    ) -> None:
        """
        This method tests the sample method.

        Args:
            shape: Shape of the weights.
            mu_prior: Mu prior value.
            std_prior: Std prior value.
            mu_init: Mu init value.
            rho_init: Rho init value.
            rngs: Random key.

        Returns:
            None.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape, mu_prior, std_prior, mu_init, rho_init, rngs
        )

        # Initial sample
        sample: jax.Array = distribution.sample()

        # Check type of sample
        assert isinstance(
            sample, jax.Array
        ), f"Incorrect type, expected {jax.Array} and got {type(sample)}"

        # Check shape of sample
        assert (
            sample.shape == shape
        ), f"Incorrect shape of sampled array, expected {shape} and got {sample.shape}"

        # Sample again
        sample_equal: jax.Array = distribution.sample(copy.deepcopy(rngs))
        sample_nequal: jax.Array = distribution.sample(copy.deepcopy(rngs))

        # Check equal samples
        assert jnp.allclose(
            sample_equal, sample_nequal
        ), "Unequal samples with same seed"

        return None

    @pytest.mark.order(3)
    def test_call(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
        rngs: Rngs,
    ) -> None:
        """
        This method tests the sample method.

        Args:
            shape: Shape of the weights.
            mu_prior: Mu prior value.
            std_prior: Std prior value.
            mu_init: Mu init value.
            rho_init: Rho init value.
            rngs: Random key.

        Returns:
            None.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape, mu_prior, std_prior, mu_init, rho_init, rngs
        )

        # Compute outputs
        outputs: jax.Array = distribution()

        # Check type of output
        assert isinstance(
            outputs, jax.Array
        ), f"Incorrect type, expected {jax.Array} and got {type(outputs)}"

        # Check outputs shape
        assert (
            outputs.shape == shape
        ), f"Incorrect shape, expected {shape} and got {outputs.shape}"

        return None

    @pytest.mark.order(4)
    def test_log_prob(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
        rngs: Rngs,
    ) -> None:
        """
        This method tests the log_prob method.

        Args:
            shape: Shape of the weights.
            mu_prior: Mu prior value.
            std_prior: Std prior value.
            mu_init: Mu init value.
            rho_init: Rho init value.
            rngs: Random key.

        Returns:
            None.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape, mu_prior, std_prior, mu_init, rho_init, rngs
        )

        # Compute log probs
        log_prob: jax.Array = distribution.log_prob()

        # Check type of output
        assert isinstance(
            log_prob, jax.Array
        ), f"Incorrect type, expected {jax.Array} and got {type(log_prob)}"

        # Check shape
        assert (
            log_prob.shape == ()
        ), f"Incorrect shape of log probs, expected {()} and got {log_prob.shape}"

        return None

    @pytest.mark.order(5)
    def test_num_params(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
        rngs: Rngs,
    ) -> None:
        """
        This method tests the sample method.

        Args:
            shape: Shape of the weights.
            mu_prior: Mu prior value.
            std_prior: Std prior value.
            mu_init: Mu init value.
            rho_init: Rho init value.
            rngs: Random key.

        Returns:
            None.
        """

        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape, mu_prior, std_prior, mu_init, rho_init, rngs
        )

        # Compute number of parameters
        num_params: int = distribution.num_params

        # Check type
        assert isinstance(
            num_params, int
        ), f"Incorrect type, expected {int} and got {type(num_params)}"

        return None
