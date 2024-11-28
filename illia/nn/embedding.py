# Libraries
from typing import Optional, Tuple, Any

from illia.distributions.dynamic import DynamicDistribution
from illia.distributions.static import StaticDistribution
from illia.nn.base import BayesianModule

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from illia.nn.torch import embedding
elif backend() == "tf":
    from illia.nn.tf import embedding


class Embedding(BayesianModule):

    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        weights_prior: Optional[StaticDistribution] = None,
        weights_posterior: Optional[DynamicDistribution] = None,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        """
        Definition of a Bayesian Embedding layer.

        Args:
            num_embeddings: Size of the dictionary of embeddings.
            embeddings_dim: The size of each embedding vector
            weights_prior: The prior distribution for the weights.
            weights_posterior: The posterior distribution for the weights.
            padding_idx: If padding_idx is specified, its entries do not affect the gradient, meaning the
                            embedding vector at padding_idx stays constant during training. Initially, this
                            embedding vector defaults to zeros but can be set to a different value to serve
                            as the padding vector.
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized to have
                        norm max_norm.
            norm_type: The p of the p-norm to compute for the max_norm option.
            scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of the words in the mini-batch.
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Call super class constructor
        super().__init__()

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

    def __call__(self, inputs: Any) -> Any:
        """
        Call the underlying layer with the given inputs to apply the layer operation.

        Args:
            inputs: The input data to the layer.

        Returns:
            The output of the layer operation.
        """

        return self.layer(inputs)

    def kl_cost(self) -> Tuple[Any, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the weights and bias of the layer.

        Returns:
            A tuple containing the KL divergence cost for the weights and bias, and the total number of parameters.
        """

        return self.layer.kl_cost()
