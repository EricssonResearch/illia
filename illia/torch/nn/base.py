# 3pp
import torch


class BayesianModule(torch.nn.Module):
    def __init__(self) -> None:
        """
        This method is the constructor for BayesianModule.
        """

        # call super class constructor
        super().__init__()

        # set state
        self.frozen: bool = False

        # create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

    @torch.jit.export
    def freeze(self) -> None:
        """
        This method freezes the layer.

        Returns:
            None.
        """

        self.frozen = True

    @torch.jit.export
    def unfreeze(self) -> None:
        """
        This method unfreezes the layer.

        Returns:
            None.
        """

        self.frozen = False

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        This is a default implementation of the kl_cots function,
        which computes

        Returns:
            tensor with the kl cost.
            number of parameters of the layer.
        """

        return torch.tensor([0.0]), 0
