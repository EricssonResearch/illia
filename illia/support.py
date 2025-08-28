"""
This module lists the supported modules and available layers for each backend.
"""

# Availables backends for DNNs and GNNs
AVAILABLE_DNN_BACKENDS: frozenset[str] = frozenset(["jax", "tf", "torch"])
AVAILABLE_GNN_BACKENDS: frozenset[str] = frozenset(["pyg"])

# Python versions to support
PYTHON_VERSIONS: list[str] = ["3.10", "3.11", "3.12"]

# PyTorch versions with the Python version to support
TORCH_COMPAT: dict[str, set[str]] = {
    "2.1.2": {"3.8", "3.9", "3.10", "3.11"},
    "2.2.2": {"3.8", "3.9", "3.10", "3.11", "3.12"},
    "2.5.1": {"3.8", "3.9", "3.10", "3.11", "3.12"},
}

# Tensorflow versions with the Python version to support
TF_COMPAT: dict[str, set[str]] = {
    "2.11.0": {"3.8", "3.9", "3.10", "3.11"},
    "2.16.1": {"3.10", "3.11", "3.12"},
    "2.19.0": {"3.10", "3.11", "3.12"},
}

# Configuration of available backends and their corresponding modules
BACKEND_MODULES: dict[str, list[str]] = {
    "torch": ["illia.nn.torch", "illia.distributions.torch", "illia.losses.torch"],
    "tf": ["illia.nn.tf", "illia.distributions.tf", "illia.losses.tf"],
    "jax": ["illia.nn.jax", "illia.distributions.jax", "illia.losses.jax"],
    "pyg": ["illia.nn.pyg"],
}

# Capabilities of each backend
BACKEND_CAPABILITIES: dict[str, dict[str, set[str]]] = {
    "torch": {
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
    "pyg": {"nn": {"CGConv"}},
}
