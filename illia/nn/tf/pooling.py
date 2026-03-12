"""TensorFlow pooling layer wrappers."""

# 3pps
from tensorflow.keras import layers


class MaxPool1d(layers.MaxPooling1D):
    """Wrapper for TensorFlow MaxPooling1D."""


class MaxPool2d(layers.MaxPooling2D):
    """Wrapper for TensorFlow MaxPooling2D."""


class AvgPool1d(layers.AveragePooling1D):
    """Wrapper for TensorFlow AveragePooling1D."""


class AvgPool2d(layers.AveragePooling2D):
    """Wrapper for TensorFlow AveragePooling2D."""


class AdaptiveAvgPool2d(layers.GlobalAveragePooling2D):
    """Wrapper for TensorFlow GlobalAveragePooling2D."""
