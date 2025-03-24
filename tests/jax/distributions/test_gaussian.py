"""
This module contains the tests for GaussianDistribution.
"""

# 3pps
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.nnx.rnglib import Rngs
import pytest

# Own modules
from illia.jax.distributions import GaussianDistribution


@pytest.mark.parametrize(
    "shape, mu_prior, std_prior, mu_init, rho_init, rngs",
    [
        ((20, 30), 0.0, 0.1, 0.0, -7.0, nnx.Rngs(0)),
        ((3, 16, 16), 0.5, 1.2, 0.5, 3.0, nnx.Rngs(1))
    ]
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
        rngs: Rngs
    ) -> None:
        
        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape, mu_prior, std_prior, mu_init, rho_init, rngs
        )
        
        # Compute number of parameters
        params = nnx.state(distribution, nnx.Param)
        num_params: int = len(params)
        
        # Check number of parameters
        assert num_params == 2, (
            f"Incorrect number of parameters, expected 2 and got {num_params}"
        )
        
        return None
    
    @pytest.mark.order(2)
    def test_sample(
        self,
        shape: tuple[int, ...],
        mu_prior: float,
        std_prior: float,
        mu_init: float,
        rho_init: float,
        rngs: Rngs
    ) -> None:
        
        # Define distribution
        distribution: GaussianDistribution = GaussianDistribution(
            shape, mu_prior, std_prior, mu_init, rho_init, rngs
        )
        
        # Initial sample
        sample: jax.Array = distribution.sample()
        
        # Check type of sample
        assert isinstance(sample, jax.Array), (
            f"Incorrect type, expected {jax.Array} and got {type(sample)}"
        )
        
        # Check shape of sample
        assert sample.shape == shape, (
            f"Incorrect shape of sampled array, expected {shape} and got {sample.shape}"
        )
        
        # Sample again
        sample_equal: jax.Array = distribution.sample(rngs)
        sample_nequal: jax.Array = distribution.sample(rngs)
        
        # Check equal samples
        assert jnp.allclose(sample, sample_equal), "Unequal samples with same seed" 
        
        return None
