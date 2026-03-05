"""
Layer definitions and mappings for different backends.
"""

# Standard libraries
from typing import Final


# Non-parametric layers by category
_POOLING_LAYERS: Final[frozenset[str]] = frozenset(
    {
        "MaxPool1d",
        "MaxPool2d",
        "AvgPool1d",
        "AvgPool2d",
        "AdaptiveAvgPool2d",
        "AdaptiveMaxPool2d",
    }
)

_ACTIVATION_LAYERS: Final[frozenset[str]] = frozenset(
    {
        "ReLU",
        "Sigmoid",
        "Tanh",
        "LeakyReLU",
        "GELU",
    }
)

_NORMALIZATION_LAYERS: Final[frozenset[str]] = frozenset(
    {
        "BatchNorm1d",
        "BatchNorm2d",
        "LayerNorm",
    }
)

_REGULARIZATION_LAYERS: Final[frozenset[str]] = frozenset(
    {
        "Dropout",
        "Dropout2d",
    }
)

_UTILITY_LAYERS: Final[frozenset[str]] = frozenset(
    {
        "Flatten",
        "Identity",
    }
)

# Bayesian layers shared across torch/tf/jax
_BAYESIAN_LAYERS: Final[frozenset[str]] = frozenset(
    {
        "BayesianModule",
        "Conv1d",
        "Conv2d",
        "Embedding",
        "Linear",
        "LSTM",
    }
)

# Mapping for non-parametric layers to native backend implementations,
# grouped by category
NONPARAMETRIC_LAYER_MAP: Final[dict[str, dict[str, dict[str, str]]]] = {
    "torch": {
        "pooling": {layer: f"torch.nn.{layer}" for layer in _POOLING_LAYERS},
        "activation": {layer: f"torch.nn.{layer}" for layer in _ACTIVATION_LAYERS},
        "normalization": {
            layer: f"torch.nn.{layer}" for layer in _NORMALIZATION_LAYERS
        },
        "regularization": {
            layer: f"torch.nn.{layer}" for layer in _REGULARIZATION_LAYERS
        },
        "utility": {layer: f"torch.nn.{layer}" for layer in _UTILITY_LAYERS},
    },
    "tf": {
        "pooling": {
            "MaxPool1d": "tensorflow.keras.layers.MaxPooling1D",
            "MaxPool2d": "tensorflow.keras.layers.MaxPooling2D",
            "AvgPool1d": "tensorflow.keras.layers.AveragePooling1D",
            "AvgPool2d": "tensorflow.keras.layers.AveragePooling2D",
            "AdaptiveAvgPool2d": "tensorflow.keras.layers.GlobalAveragePooling2D",
        },
        "activation": {
            "ReLU": "tensorflow.keras.layers.Activation",
            "Sigmoid": "tensorflow.keras.layers.Activation",
            "Tanh": "tensorflow.keras.layers.Activation",
            "LeakyReLU": "tensorflow.keras.layers.LeakyReLU",  # HACK: diff name func
            "GELU": "tensorflow.keras.layers.Activation",
        },
        "normalization": {
            "BatchNorm1d": "tensorflow.keras.layers.BatchNormalization",
            "BatchNorm2d": "tensorflow.keras.layers.BatchNormalization",
            "LayerNorm": "tensorflow.keras.layers.LayerNormalization",
        },
        "regularization": {
            "Dropout": "tensorflow.keras.layers.Dropout",
            "Dropout2d": "tensorflow.keras.layers.SpatialDropout2D",
        },
        "utility": {
            "Flatten": "tensorflow.keras.layers.Flatten",
        },
    },
    "jax": {
        "pooling": {
            "MaxPool2d": "flax.nnx.max_pool",  # TBD: name 2d?: Y/N
            "AvgPool2d": "flax.nnx.avg_pool",
        },
        "activation": {
            "ReLU": "flax.nnx.relu",
            "Sigmoid": "flax.nnx.sigmoid",
            "Tanh": "flax.nnx.tanh",
            "GELU": "flax.nnx.gelu",
        },
        "normalization": {
            "BatchNorm2d": "flax.nnx.BatchNorm",  # TBD: name 2d?: Y/N
            "LayerNorm": "flax.nnx.LayerNorm",
        },
        "regularization": {
            "Dropout": "flax.nnx.Dropout",
        },
    },
}

# Flat mapping
LAYER_CATEGORY_MAP: Final[dict[str, str]] = {
    **{layer: "pooling" for layer in _POOLING_LAYERS},
    **{layer: "activation" for layer in _ACTIVATION_LAYERS},
    **{layer: "normalization" for layer in _NORMALIZATION_LAYERS},
    **{layer: "regularization" for layer in _REGULARIZATION_LAYERS},
    **{layer: "utility" for layer in _UTILITY_LAYERS},
}
