"""JAX regularization layer wrappers."""

# 3pps
from flax import nnx


class Dropout:
    """Wrapper for JAX Dropout."""

    def __new__(cls, *args, **kwargs):
        return nnx.Dropout(*args, **kwargs)
