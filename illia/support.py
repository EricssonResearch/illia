"""
This module defines supported deep-learning and graph-learning backends,
compatible versions, and the available layers/modules for each backend.
It serves as a central configuration for backend-specific capabilities.
"""

# Standard libraries
from typing import Final


# Name of the environment variable to switch between backends at runtime
ENV_OS_NAME: Final[str] = "ILLIA_BACKEND"

# Supported Deep Neural Network (DNN) backends
AVAILABLE_DNN_BACKENDS: frozenset[str] = frozenset(["jax", "tf", "torch"])

# Supported Graph Neural Network (GNN) backends
AVAILABLE_GNN_BACKENDS: frozenset[str] = frozenset(["pyg"])

# Default backend if none is specified in the environment or configuration
DEFAULT_BACKEND: Final[str] = "torch"

# Supported Python versions for the project
PYTHON_VERSIONS: Final[tuple[str, ...]] = ("3.10", "3.11", "3.12")

# PyTorch versions and their compatible Python versions
TORCH_COMPAT: Final[dict[str, set[str]]] = {
    "2.1.2": {"3.8", "3.9", "3.10", "3.11"},
    "2.2.2": {"3.8", "3.9", "3.10", "3.11", "3.12"},
    "2.5.1": {"3.8", "3.9", "3.10", "3.11", "3.12"},
}

# TensorFlow versions and their compatible Python versions
TF_COMPAT: Final[dict[str, set[str]]] = {
    "2.11.0": {"3.8", "3.9", "3.10", "3.11"},
    "2.16.1": {"3.10", "3.11", "3.12"},
    "2.19.0": {"3.10", "3.11", "3.12"},
}

# Mapping of each backend to the list of Python modules it provides
BACKEND_MODULES: Final[dict[str, list[str]]] = {
    "torch": ["illia.nn.torch", "illia.distributions.torch", "illia.losses.torch"],
    "tf": ["illia.nn.tf", "illia.distributions.tf", "illia.losses.tf"],
    "jax": ["illia.nn.jax", "illia.distributions.jax", "illia.losses.jax"],
    "pyg": ["illia.nn.pyg"],
}

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

# Dictionary describing the layers and capabilities supported by each backend
BACKEND_CAPABILITIES: Final[dict[str, dict[str, set[str]]]] = {
    "torch": {
        "nn": {
            *_BAYESIAN_LAYERS,
            *_POOLING_LAYERS,
            *_ACTIVATION_LAYERS,
            *_NORMALIZATION_LAYERS,
            *_REGULARIZATION_LAYERS,
            *_UTILITY_LAYERS,
        },
        "distributions": {"DistributionModule", "GaussianDistribution"},
        "losses": {"KLDivergenceLoss", "ELBOLoss"},
    },
    "tf": {
        "nn": {
            *_BAYESIAN_LAYERS,
            # Pooling (except AdaptiveMaxPool2d)
            "MaxPool1d",
            "MaxPool2d",
            "AvgPool1d",
            "AvgPool2d",
            "AdaptiveAvgPool2d",
            # Activations (except GELU in older TF)
            "ReLU",
            "Sigmoid",
            "Tanh",
            "LeakyReLU",
            "GELU",
            *_NORMALIZATION_LAYERS,
            *_REGULARIZATION_LAYERS,
            "Flatten",  # No Identity in TF
        },
        "distributions": {"DistributionModule", "GaussianDistribution"},
        "losses": {"KLDivergenceLoss", "ELBOLoss"},
    },
    "jax": {
        "nn": {
            *_BAYESIAN_LAYERS,
            # JAX limited non-parametric layers
            "MaxPool2d",
            "AvgPool2d",
            "ReLU",
            "Sigmoid",
            "Tanh",
            "GELU",
            "BatchNorm2d",
            "LayerNorm",
            "Dropout",
        },
        "distributions": {"DistributionModule", "GaussianDistribution"},
        "losses": {"KLDivergenceLoss", "ELBOLoss"},
    },
    "pyg": {
        "nn": {"CGConv"},
    },
}

# HACK: risk of path hard-coding.
# Mapping for non-parametric layers to native backend implementations
NONPARAMETRIC_LAYER_MAP: Final[dict[str, dict[str, str]]] = {
    "torch": {
        **{layer: f"torch.nn.{layer}" for layer in _POOLING_LAYERS},
        **{layer: f"torch.nn.{layer}" for layer in _ACTIVATION_LAYERS},
        **{layer: f"torch.nn.{layer}" for layer in _NORMALIZATION_LAYERS},
        **{layer: f"torch.nn.{layer}" for layer in _REGULARIZATION_LAYERS},
        **{layer: f"torch.nn.{layer}" for layer in _UTILITY_LAYERS},
    },
    "tf": {
        # TODO: decide: import from tf or update to keras only.
        # Pooling : diff naming convention
        "MaxPool1d": "tensorflow.keras.layers.MaxPooling1D",
        "MaxPool2d": "tensorflow.keras.layers.MaxPooling2D",
        "AvgPool1d": "tensorflow.keras.layers.AveragePooling1D",
        "AvgPool2d": "tensorflow.keras.layers.AveragePooling2D",
        "AdaptiveAvgPool2d": "tensorflow.keras.layers.GlobalAveragePooling2D",
        # Activations
        "ReLU": "tensorflow.keras.layers.Activation",
        "Sigmoid": "tensorflow.keras.layers.Activation",
        "Tanh": "tensorflow.keras.layers.Activation",
        "LeakyReLU": "tensorflow.keras.layers.LeakyReLU",  # HACK: name inconsistency
        "GELU" : "tensorflow.keras.layers.Activation",
        # Normalization
        "BatchNorm1d": "tensorflow.keras.layers.BatchNormalization",
        "BatchNorm2d": "tensorflow.keras.layers.BatchNormalization",
        "BatchNorm3d": "tensorflow.keras.layers.BatchNormalization",
        "LayerNorm": "tensorflow.keras.layers.LayerNormalization",
        # Regularization
        "Dropout": "tensorflow.keras.layers.Dropout",
        "Dropout2d": "tensorflow.keras.layers.SpatialDropout2D",
        # Utility
        "Flatten": "tensorflow.keras.layers.Flatten",
    },
    "jax": {
        # JAX/Flax diff naming
        "MaxPool2d": "flax.linen.max_pool",
        "AvgPool2d": "flax.linen.avg_pool",
        "ReLU": "flax.linen.relu",
        "Sigmoid": "flax.linen.sigmoid",
        "Tanh": "flax.linen.tanh",
        "GELU": "flax.linen.gelu",
        "BatchNorm2d": "flax.linen.BatchNorm",
        "LayerNorm": "flax.linen.LayerNorm",
        "Dropout": "flax.linen.Dropout",
    },
}
