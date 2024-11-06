# Libraries
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

from illia.distributions import dynamic
from illia.distributions import static


class Embedding(ABC):

    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        weights_prior: Optional[static.StaticDistribution] = None,
        weights_posterior: Optional[dynamic.DynamicDistribution] = None,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        Definition of a Bayesian Embedding layer.

        Args:
            num_embeddings (int): Size of the dictionary of embeddings.
            embeddings_dim (int): The size of each embedding vector
            weights_prior (Optional[StaticDistribution], optional): The prior distribution for the weights. Defaults to None.
            weights_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the weights. Defaults to None.
            padding_idx (Optional[int], optional): If padding_idx is specified, its entries do not affect the gradient, meaning the 
                                                    embedding vector at padding_idx stays constant during training. Initially, this 
                                                    embedding vector defaults to zeros but can be set to a different value to serve 
                                                    as the padding vector.
            max_norm (Optional[float], optional): If given, each embedding vector with norm larger than max_norm is renormalized to have 
                                                    norm max_norm. Defaults to None.
            norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Defaults to 2.0.
            scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Defaults to False.
            sparse (bool, optional): If True, gradient w.r.t. weight matrix will be a sparse tensor. Defaults to False.
            backend (Optional[str], optional): The backend to use. Defaults to 'torch'.

        Raises:
            ValueError: If an invalid backend value is provided.
        """
        
        # Call super class constructor
        super(Embedding, self).__init__()

        # Set attributes
        self.backend = backend

        # Choose backend
        if self.backend == "torch":
            # Import torch part
            from illia.nn.torch import embedding
        elif self.backend == "tf":
            # Import tensorflow part
            from illia.nn.tf import embedding
        else:
            raise ValueError("Invalid backend value")

        # Define layer based on the imported library
        self.layer = embedding.Embedding(
            num_embeddings=num_embeddings,
            embeddings_dim=embeddings_dim,
            weights_prior=weights_prior,
            weights_posterior=weights_posterior,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    def __call__(self, inputs):
        return self.layer(inputs)

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        pass
