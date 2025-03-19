# standard libraries
from abc import abstractmethod

# 3pp
import jax
from flax import nnx


class BayesianModule(nnx.Module):
    """
    This class is the base class for BayesianModule for jax. Every
    module that should behave as a bayesian layer should inherit from
    this class.

    Attr:
        frozen: indicator if this module if frozen or not.
    """

    frozen: bool

    def __init__(self) -> None:
        # set freeze false by default
        self.frozen = False

    def freeze(self) -> None:
        # set frozen indicator to true for current layer
        self.frozen = True

        # set forzen indicator to true for children
        for _, module in self.iter_modules():
            if self != module and isinstance(module, BayesianModule):
                module.freeze()
            else:
                continue

    def unfreeze(self) -> None:
        # set frozen indicator to false for current layer
        self.frozen = False

        # set forzen indicators to false for children
        for _, module in self.iter_modules():
            if module != self and isinstance(module, BayesianModule):
                module.unfreeze()

    @abstractmethod
    def kl_cost(self) -> tuple[jax.Array, int]:
        pass
