# standard libraries
from typing import Optional

# 3pp
import torch
import torch.nn.functional as F

# own modules
from illia.torch.nn.base import BayesianModule
from illia.torch.distributions import (
    Distribution,
    GaussianDistribution,
)


class Embedding(BayesianModule):
    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        weights_distribution: Optional[Distribution] = None,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        # call super class constructor
        super().__init__()

        # set weights distribution
        self.weights_distribution: Distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                (num_embeddings, embeddings_dim)
            )
        else:
            self.weights_distribution = weights_distribution

        # sample initial weights
        weights = self.weights_distribution.sample()

        # register buffers
        self.register_buffer("weights", weights)

        # set embeddings atributtes
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()

        else:
            if self.weights is None:
                raise ValueError("Module has been frozen with undefined weights")

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

    @torch.jit.export
    def freeze(self) -> None:
        # set indicator
        self.frozen = True

        # detach weights
        self.weights = self.weights.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        # get log posterior and log prior
        log_probs: torch.Tensor = self.weights_distribution.log_prob(self.weights)

        # get number of parameters
        num_params: int = self.weights_distribution.num_params

        return log_probs, num_params
