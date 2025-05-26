"""
This module lists the supported modules and available layers for each backend.
"""

# Source of Illia versioning
VERSION: str = "0.0.1"

# Configuration of available backends and its corresponding modules
BACKEND_MODULES: dict[str, list[str]] = {
    "torch": ["illia.nn.torch", "illia.distributions.torch", "illia.losses.torch"],
    "tf": ["illia.nn.tf", "illia.distributions.tf", "illia.losses.tf"],
    "jax": ["illia.nn.jax", "illia.distributions.jax"],
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
        "nn": {"BayesianModule", "Linear"},
        "distributions": {"DistributionModule", "GaussianDistribution"},
    },
    "pyg": {"nn": {"CGConv"}},
}
