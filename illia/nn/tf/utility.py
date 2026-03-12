"""TensorFlow utility layer wrappers."""

# 3pps
from tensorflow.keras import layers


class Flatten(layers.Flatten):
    """Wrapper for TensorFlow Flatten."""
