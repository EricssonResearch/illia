"""TensorFlow regularization layer wrappers."""

# 3pps
from tensorflow.keras import layers


class Dropout(layers.Dropout):
    """Wrapper for TensorFlow Dropout."""


class Dropout2d(layers.SpatialDropout2D):
    """Wrapper for TensorFlow SpatialDropout2D."""
