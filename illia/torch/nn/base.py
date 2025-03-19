# Libraries
import torch


class BayesianModule(torch.nn.Module):
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
        self.frozen: bool = False
        
        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    @torch.jit.export
    def unfreeze(self) -> None:
        """
        Unfreezes the current module and all submodules that are
        instances of BayesianModule. Sets the frozen state to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Abstract method to compute the KL divergence cost.
        Must be implemented by subclasses.

        Returns:
            A tuple containing the KL divergence cost and its
            associated integer value.
        """

        return torch.tensor([0.0]), 0