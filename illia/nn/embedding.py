# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Optional, Tuple

# own modules
import torch_bayesian.distributions.static as static
from torch_bayesian.nn.base import BayesianModule
from torch_bayesian.distributions.static import StaticDistribution
from torch_bayesian.distributions.dynamic import (
    DynamicDistribution,
    GaussianDistribution,
)


class Embedding(BayesianModule):
    input_size: int
    output_size: int
    weights_posterior: DynamicDistribution
    weights_prior: StaticDistribution
    bias_posterior: DynamicDistribution
    bias_prior: StaticDistribution
    weights: torch.Tensor
    bias: torch.Tensor

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
        # call super class constructor
        super().__init__()

        # define default parameters
        parameters = {"mean": 0, "std": 0.1}

        # set prior if they are None
        if weights_prior is None:
            self.weights_prior = static.GaussianDistribution(parameters)
        else:
            self.weights_prior = weights_prior

        if weights_posterior is None:
            self.weights_posterior = GaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_posterior = weights_posterior

        # set embeddings atributtes
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()

        else:
            if self.weights is None or self.bias is None:
                self.weights = self.weights_posterior.sample()
                self.bias = self.bias_posterior.sample()

        # run torch forward
        outputs: torch.Tensor = F.embedding(
            inputs,
            self.weights,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return outputs

    def kl_cost(self) -> Tuple[torch.Tensor, int]:
        # get log posterior and log prior
        log_posterior: torch.Tensor = self.weights_posterior.log_prob(
            self.weights
        ) + self.bias_posterior.log_prob(self.bias)
        log_prior: torch.Tensor = self.weights_prior.log_prob(
            self.weights
        ) + self.bias_prior.log_prob(self.bias)

        # get number of parameters
        num_params: int = (
            self.weights_posterior.num_params + self.bias_posterior.num_params
        )

        return log_posterior - log_prior, num_params
