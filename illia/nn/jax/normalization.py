"""JAX normalization layer wrappers."""

# 3pps
from flax import nnx


class BatchNorm1d:
    """Wrapper for JAX BatchNorm for 1D inputs."""

    def __new__(cls, num_features, *args, **kwargs):
        return nnx.BatchNorm(num_features, *args, **kwargs)


class BatchNorm2d:
    """Wrapper for JAX BatchNorm for 2D inputs."""

    def __new__(cls, num_features, *args, **kwargs):
        return nnx.BatchNorm(num_features, *args, **kwargs)


class LayerNorm:
    """Wrapper for JAX LayerNorm."""

    def __new__(cls, *args, **kwargs):
        return nnx.LayerNorm(*args, **kwargs)
