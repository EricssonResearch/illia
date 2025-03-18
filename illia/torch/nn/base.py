# Libraries
from abc import abstractmethod
from typing import Tuple, Any

from torch.nn import Module


class BayesianModule(Module):
    """
    Base class for creating a Bayesian module, which can be frozen or
    unfrozen. This class is intended to be subclassed for specific
    backend implementations.
    """

    frozen: bool

    def __init__(self):
        """
        Initializes the BayesianModule, setting the frozen state to
        False.
        """

        # Call super class constructor
        super().__init__()

        # Set freeze false by default
        self.frozen = False

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

        # Set frozen indicator to true for children
        for module in self.modules():
            if self != module and isinstance(module, BayesianModule):
                module.freeze()
            else:
                continue

    def unfreeze(self) -> None:
        """
        Unfreezes the current module and all submodules that are
        instances of BayesianModule. Sets the frozen state to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

        # Set frozen indicators to false for children
        for module in self.modules():
            if self != module and isinstance(module, BayesianModule):
                module.unfreeze()
            else:
                continue

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        """
        Abstract method to compute the KL divergence cost.
        Must be implemented by subclasses.

        Returns:
            A tuple containing the KL divergence cost and its
            associated integer value.
        """
