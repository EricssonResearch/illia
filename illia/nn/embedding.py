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
            num_embeddings (int): _description_
            embeddings_dim (int): _description_
            weights_prior (Optional[static.StaticDistribution], optional): _description_. Defaults to None.
            weights_posterior (Optional[dynamic.DynamicDistribution], optional): _description_. Defaults to None.
            padding_idx (Optional[int], optional): _description_. Defaults to None.
            max_norm (Optional[float], optional): _description_. Defaults to None.
            norm_type (float, optional): _description_. Defaults to 2.0.
            scale_grad_by_freq (bool, optional): _description_. Defaults to False.
            sparse (bool, optional): _description_. Defaults to False.
            backend (Optional[str], optional): _description_. Defaults to "torch".

        Raises:
            ValueError: _description_
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
