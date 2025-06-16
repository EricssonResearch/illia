"""
This module contains auxiliary functions for jax.nn.
"""

# 3pps
from flax import nnx


def freeze(model: nnx.Module) -> None:
    """
    This function freezes all bayesian layers of a model. It will
    modify the model that is passed as an argument.

    Args:
        model: pytorch model.
    """

    for _, module in model.iter_modules():
        if hasattr(module, "is_bayesian"):
            module.freeze()  # type: ignore


def unfreeze(model: nnx.Module) -> None:
    """
    This function freezes all bayesian layers of a model. It will
    modify the model that is passed as an argument.

    Args:
        model: pytorch model.
    """

    for _, module in model.iter_modules():
        if hasattr(module, "is_bayesian"):
            module.unfreeze()  # type: ignore
