"""TensorFlow activation layer wrappers."""

# 3pps
from tensorflow.keras import layers


class ReLU(layers.Activation):
    """Wrapper for TensorFlow ReLU activation."""

    def __init__(self, *args, **kwargs):
        super().__init__("relu", *args, **kwargs)


class Sigmoid(layers.Activation):
    """Wrapper for TensorFlow Sigmoid activation."""

    def __init__(self, *args, **kwargs):
        super().__init__("sigmoid", *args, **kwargs)


class Tanh(layers.Activation):
    """Wrapper for TensorFlow Tanh activation."""

    def __init__(self, *args, **kwargs):
        super().__init__("tanh", *args, **kwargs)


class LeakyReLU(layers.Activation):
    """Wrapper for TensorFlow LeakyReLU."""

    def __init__(self, *args, **kwargs):
        super().__init__("leaky_relu", *args, **kwargs)


class GELU(layers.Activation):
    """Wrapper for TensorFlow GELU activation."""

    def __init__(self, *args, **kwargs):
        super().__init__("gelu", *args, **kwargs)
