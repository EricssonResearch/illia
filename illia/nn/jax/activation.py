"""JAX activation layer wrappers."""

# Standard libraries
from functools import partial

# 3pps
from flax import nnx


class ReLU:
    """Wrapper for JAX relu."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.relu, *args, **kwargs)


class Sigmoid:
    """Wrapper for JAX sigmoid."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.sigmoid, *args, **kwargs)


class Tanh:
    """Wrapper for JAX tanh."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.tanh, *args, **kwargs)


class GELU:
    """Wrapper for JAX gelu."""

    def __new__(cls, *args, **kwargs):
        return partial(nnx.gelu, *args, **kwargs)
