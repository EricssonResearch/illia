"""
This module defines supported deep-learning and graph-learning backends,
compatible versions, and the available layers/modules for each backend.
It serves as a central configuration for backend-specific capabilities.
"""

# Name of the environment variable to switch between backends at runtime
ENV_OS_NAME: str = "ILLIA_BACKEND"

# Supported Deep Neural Network (DNN) backends
AVAILABLE_DNN_BACKENDS: frozenset[str] = frozenset(["jax", "tf", "torch"])

# Supported Graph Neural Network (GNN) backends
AVAILABLE_GNN_BACKENDS: frozenset[str] = frozenset(["pyg"])

# Default backend if none is specified in the environment or configuration
DEFAULT_BACKEND: str = "torch"

# Supported Python versions for the project
PYTHON_VERSIONS: list[str] = ["3.10", "3.11", "3.12"]

# PyTorch versions and their compatible Python versions
TORCH_COMPAT: dict[str, set[str]] = {
    "2.1.2": {"3.8", "3.9", "3.10", "3.11"},
    "2.2.2": {"3.8", "3.9", "3.10", "3.11", "3.12"},
    "2.5.1": {"3.8", "3.9", "3.10", "3.11", "3.12"},
}

# TensorFlow versions and their compatible Python versions
TF_COMPAT: dict[str, set[str]] = {
    "2.11.0": {"3.8", "3.9", "3.10", "3.11"},
    "2.16.1": {"3.10", "3.11", "3.12"},
    "2.19.0": {"3.10", "3.11", "3.12"},
}

# Mapping of each backend to the list of Python modules it provides
BACKEND_MODULES: dict[str, list[str]] = {
    "torch": ["illia.nn.torch", "illia.distributions.torch", "illia.losses.torch"],
    "tf": ["illia.nn.tf", "illia.distributions.tf", "illia.losses.tf"],
    "jax": ["illia.nn.jax", "illia.distributions.jax", "illia.losses.jax"],
    "pyg": ["illia.nn.pyg"],
}

# Dictionary describing the layers and capabilities supported by each backend
BACKEND_CAPABILITIES: dict[str, dict[str, set[str]]] = {
    "torch": {
        "nn": {"BayesianModule", "Conv1D", "Conv2D", "Embedding", "Linear", "LSTM"},
        "distributions": {"DistributionModule", "GaussianDistribution"},
        "losses": {
            "KLDivergenceLoss",
            "ELBOLoss",
        },
    },
    "tf": {
        "nn": {
            "BayesianModule",
            "Conv1D",
            "Conv2D",
            "Embedding",
            "Linear",
        },
        "distributions": {"DistributionModule", "GaussianDistribution"},
        "losses": {
            "KLDivergenceLoss",
            "ELBOLoss",
        },
    },
    "jax": {
        "nn": {
            "BayesianModule",
            "Conv1D",
            "Conv2D",
            "Embedding",
            "Linear",
        },
        "distributions": {"DistributionModule", "GaussianDistribution"},
        "losses": {
            "KLDivergenceLoss",
            "ELBOLoss",
        },
    },
    "pyg": {
        "nn": {"CGConv"},
    },
}
