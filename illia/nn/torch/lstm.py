# Standard libraries
from typing import Any, Optional

# 3pps
import torch

# Own modules
from illia.distributions.torch.gaussian import GaussianDistribution
from illia.nn.torch.base import BayesianModule
from illia.nn.torch.embedding import Embedding


class LSTM(BayesianModule):
    """
    Bayesian LSTM layer with embedding and probabilistic weights.
    All weights and biases are sampled from Gaussian distributions.
    Freezing the layer fixes parameters and stops gradient computation.
    """

    # Forget gate
    wf: torch.Tensor
    bf: torch.Tensor

    # Input gate
    wi: torch.Tensor
    bi: torch.Tensor

    # Candidate gate
    wc: torch.Tensor
    bc: torch.Tensor

    # Output gate
    wo: torch.Tensor
    bo: torch.Tensor

    # Final output layer
    wv: torch.Tensor
    bv: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        hidden_size: int,
        output_size: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Bayesian LSTM layer.

        Args:
            num_embeddings: Size of the embedding dictionary.
            embeddings_dim: Dimensionality of each embedding vector.
            hidden_size: Number of hidden units in the LSTM.
            output_size: Size of the final output.
            padding_idx: Index to ignore in embeddings.
            max_norm: Maximum norm for embedding vectors.
            norm_type: Norm type used for max_norm.
            scale_grad_by_freq: Scale gradient by inverse frequency.
            sparse: Use sparse embedding updates.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        super().__init__(**kwargs)

        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Define the Embedding layer
        self.embedding = Embedding(
            num_embeddings=self.num_embeddings,
            embeddings_dim=self.embeddings_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Initialize weights
        # Forget gate
        self.wf_distribution = GaussianDistribution(
            (self.hidden_size, self.embeddings_dim + self.hidden_size)
        )
        self.bf_distribution = GaussianDistribution((self.hidden_size,))

        # Input gate
        self.wi_distribution = GaussianDistribution(
            (self.hidden_size, self.embeddings_dim + self.hidden_size)
        )
        self.bi_distribution = GaussianDistribution((self.hidden_size,))

        # Candidate gate
        self.wc_distribution = GaussianDistribution(
            (self.hidden_size, self.embeddings_dim + self.hidden_size)
        )
        self.bc_distribution = GaussianDistribution((self.hidden_size,))

        # Output gate
        self.wo_distribution = GaussianDistribution(
            (self.hidden_size, self.embeddings_dim + self.hidden_size)
        )
        self.bo_distribution = GaussianDistribution((self.hidden_size,))

        # Final gate
        self.wv_distribution = GaussianDistribution(
            (self.output_size, self.hidden_size)
        )
        self.bv_distribution = GaussianDistribution((self.output_size,))

        # Sample initial weights and register buffers
        # Forget gate
        wf = self.wf_distribution.sample()
        bf = self.bf_distribution.sample()
        self.register_buffer("wf", wf)
        self.register_buffer("bf", bf)

        # Input gate
        wi = self.wi_distribution.sample()
        bi = self.bi_distribution.sample()
        self.register_buffer("wi", wi)
        self.register_buffer("bi", bi)

        # Candidate gate
        wc = self.wc_distribution.sample()
        bc = self.bc_distribution.sample()
        self.register_buffer("wc", wc)
        self.register_buffer("bc", bc)

        # Output gate
        wo = self.wo_distribution.sample()
        bo = self.bo_distribution.sample()
        self.register_buffer("wo", wo)
        self.register_buffer("bo", bo)

        # Final output layer
        wv = self.wv_distribution.sample()
        bv = self.bv_distribution.sample()
        self.register_buffer("wv", wv)
        self.register_buffer("bv", bv)

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Freeze embedding layer
        self.embedding.freeze()

        # Forget gate
        if self.wf is None:
            self.wf = self.wf_distribution.sample()
        if self.bf is None:
            self.bf = self.bf_distribution.sample()
        self.wf = self.wf.detach()
        self.bf = self.bf.detach()

        # Input gate
        if self.wi is None:
            self.wi = self.wi_distribution.sample()
        if self.bi is None:
            self.bi = self.bi_distribution.sample()
        self.wi = self.wi.detach()
        self.bi = self.bi.detach()

        # Candidate gate
        if self.wc is None:
            self.wc = self.wc_distribution.sample()
        if self.bc is None:
            self.bc = self.bc_distribution.sample()
        self.wc = self.wc.detach()
        self.bc = self.bc.detach()

        # Output gate
        if self.wo is None:
            self.wo = self.wo_distribution.sample()
        if self.bo is None:
            self.bo = self.bo_distribution.sample()
        self.wo = self.wo.detach()
        self.bo = self.bo.detach()

        # Final output layer
        if self.wv is None:
            self.wv = self.wv_distribution.sample()
        if self.bv is None:
            self.bv = self.bv_distribution.sample()
        self.wv = self.wv.detach()
        self.bv = self.bv.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the KL
                divergence cost and the total number of parameters in
                the layer.
        """

        # Compute log probs for each pair of weights and bias
        # Forget gate
        log_probs_f: torch.Tensor = self.wf_distribution.log_prob(
            self.wf
        ) + self.bf_distribution.log_prob(self.bf)
        # Input gate
        log_probs_i: torch.Tensor = self.wi_distribution.log_prob(
            self.wi
        ) + self.bi_distribution.log_prob(self.bi)
        # Candidate gate
        log_probs_c: torch.Tensor = self.wc_distribution.log_prob(
            self.wc
        ) + self.bc_distribution.log_prob(self.bc)
        # Output gate
        log_probs_o: torch.Tensor = self.wo_distribution.log_prob(
            self.wo
        ) + self.bo_distribution.log_prob(self.bo)
        # Final output layer
        log_probs_v: torch.Tensor = self.wv_distribution.log_prob(
            self.wv
        ) + self.bv_distribution.log_prob(self.bv)

        # Compute the total loss
        log_probs = log_probs_f + log_probs_i + log_probs_c + log_probs_o + log_probs_v

        # Compute number of parameters
        num_params: int = (
            self.wf_distribution.num_params()
            + self.bf_distribution.num_params()
            + self.wi_distribution.num_params()
            + self.bi_distribution.num_params()
            + self.wc_distribution.num_params()
            + self.bc_distribution.num_params()
            + self.wo_distribution.num_params()
            + self.bo_distribution.num_params()
            + self.wv_distribution.num_params()
            + self.bv_distribution.num_params()
        )

        return log_probs, num_params

    def forward(
        self,
        inputs: torch.Tensor,
        init_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a forward pass through the Bayesian LSTM layer.
        If the layer is not frozen, it samples weights and bias
        from their respective distributions. If the layer is frozen
        and the weights or bias are not initialized, it also performs
        sampling.

        Args:
            inputs: Input tensor to the layer with shape [batch,
                input channels, input width, input height].

        Returns:
            Output tensor after passing through the layer with shape
                [batch, output channels, output width, output height].

        Raises:
            ValueError: If the layer is frozen but weights are
                undefined.
        """

        # Sample weights if not frozen
        if not self.frozen:
            self.wf = self.wf_distribution.sample()
            self.bf = self.bf_distribution.sample()
            self.wi = self.wi_distribution.sample()
            self.bi = self.bi_distribution.sample()
            self.wc = self.wc_distribution.sample()
            self.bc = self.bc_distribution.sample()
            self.wo = self.wo_distribution.sample()
            self.bo = self.bo_distribution.sample()
            self.wv = self.wv_distribution.sample()
            self.bv = self.bv_distribution.sample()
        elif any(
            p is None
            for p in [
                self.wf,
                self.bf,
                self.wi,
                self.bi,
                self.wc,
                self.bc,
                self.wo,
                self.bo,
                self.wv,
                self.bv,
            ]
        ):
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Apply embedding layer to input indices
        inputs = inputs.squeeze(dim=-1)
        inputs = self.embedding(inputs)
        batch_size, seq_len, _ = inputs.size()

        # Initialize h_t and c_t if init_states is None
        if init_states is None:
            device = inputs.device
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_states[0], init_states[1]

        for t in range(seq_len):
            # Shape: (batch_size, embedding_dim)
            x_t = inputs[:, t, :]

            # Concatenate input and hidden state
            # Shape: (batch_size, embedding_dim + hidden_size)
            z_t = torch.cat([x_t, h_t], dim=1)

            # Forget gate
            ft = torch.sigmoid(z_t @ self.wf.t() + self.bf)

            # Input gate
            it = torch.sigmoid(z_t @ self.wi.t() + self.bi)

            # Candidate cell state
            can = torch.tanh(z_t @ self.wc.t() + self.bc)

            # Output gate
            ot = torch.sigmoid(z_t @ self.wo.t() + self.bo)

            # Update cell state
            c_t = c_t * ft + can * it

            # Update hidden state
            h_t = ot * torch.tanh(c_t)

        # Compute final output
        y_t = h_t @ self.wv.t() + self.bv

        return y_t, (h_t, c_t)
