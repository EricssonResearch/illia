"""JAX pooling layer wrappers."""

# Standard libraries
from functools import partial

# 3pps
import flax.nnx as nnx


class MaxPool1d:
    """Wrapper for JAX max_pool with 1D window."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.max_pool, *args, **kwargs)


class MaxPool2d:
    """Wrapper for JAX max_pool with 2D window."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.max_pool, *args, **kwargs)


class AvgPool1d:
    """Wrapper for JAX avg_pool with 1D window."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.avg_pool, *args, **kwargs)


class AvgPool2d:
    """Wrapper for JAX avg_pool with 2D window."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.avg_pool, *args, **kwargs)