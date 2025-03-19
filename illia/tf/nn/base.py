# Libraries
import tensorflow as tf
from keras import Model, saving


class BayesianModule(Model):
    """
    Base class for creating a Bayesian module, which can be frozen or
    unfrozen. This class is intended to be subclassed for specific
    backend implementations.
    """
    
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

    def freeze(self) -> None:
        """
        Freezes the current layer and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    def unfreeze(self) -> None:
        """
        Unfreezes the current layer and all submodules that are
        instances of BayesianModule. Sets the frozen state to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Abstract method to compute the KL divergence cost.
        Must be implemented by subclasses.

        Returns:
            A tuple containing the KL divergence cost and its
            associated integer value.
        """

        return tf.Tensor([0.0]), 0