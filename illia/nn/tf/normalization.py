"""TensorFlow normalization layer wrappers."""

# 3pps
from tensorflow.keras import layers


class BatchNorm1d(layers.BatchNormalization):
    """Wrapper for TensorFlow BatchNormalization (1D)."""


class BatchNorm2d(layers.BatchNormalization):
    """Wrapper for TensorFlow BatchNormalization (2D)."""


class LayerNorm(layers.LayerNormalization):
    """Wrapper for TensorFlow LayerNormalization."""
