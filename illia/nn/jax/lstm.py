"""
This module contains the code for the bayesian LSTM.
"""

# Standard libraries
from typing import Any, Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs

# Own modules
from illia.distributions.jax.gaussian import GaussianDistribution
from illia.nn.jax.base import BayesianModule
from illia.nn.jax.embedding import Embedding


class LSTM(BayesianModule):
    """
    This class is the bayesian implementation of the TensorFlow LSTM layer.
    """

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
        rngs: Rngs = nnx.Rngs(0),
        **kwargs: Any,
    ) -> None:
        """_summary_

        Args:
            num_embeddings (int): _description_
            embeddings_dim (int): _description_
            hidden_size (int): _description_
            output_size (int): _description_
            padding_idx (Optional[int], optional): _description_. Defaults to None.
            max_norm (Optional[float], optional): _description_. Defaults to None.
            norm_type (float, optional): _description_. Defaults to 2.0.
            scale_grad_by_freq (bool, optional): _description_. Defaults to False.
            rngs (Rngs, optional): _description_. Defaults to nnx.Rngs(0).

        Returns:
            None.
        """

        # Call super-class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.rngs = rngs

        # Define the Embedding layer
        self.embedding = Embedding(
            num_embeddings=self.num_embeddings,
            embeddings_dim=self.embeddings_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            rngs=self.rngs,
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
        self.wf = nnx.Param(self.wf_distribution.sample(self.rngs))
        self.bf = nnx.Param(self.bf_distribution.sample(self.rngs))

        # Input gate
        self.wi = nnx.Param(self.wi_distribution.sample(self.rngs))
        self.bi = nnx.Param(self.bi_distribution.sample(self.rngs))

        # Candidate gate
        self.wc = nnx.Param(self.wc_distribution.sample(self.rngs))
        self.bc = nnx.Param(self.bc_distribution.sample(self.rngs))

        # Output gate
        self.wo = nnx.Param(self.wo_distribution.sample(self.rngs))
        self.bo = nnx.Param(self.bo_distribution.sample(self.rngs))

        # Final output layer
        self.wv = nnx.Param(self.wv_distribution.sample(self.rngs))
        self.bv = nnx.Param(self.bv_distribution.sample(self.rngs))

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.

        Returns:
            None.
        """
        
        # Set indicator
        self.frozen = True

        # Freeze embedding layer
        self.embedding.freeze()

        # Forget gate
        if self.wf is None:
            self.wf = nnx.Param(self.wf_distribution.sample(self.rngs))
        if self.bf is None:
            self.bf = nnx.Param(self.bf_distribution.sample(self.rngs))
        self.wf = jax.lax.stop_gradient(self.wf)
        self.bf = jax.lax.stop_gradient(self.bf)

        # Input gate
        if self.wi is None:
            self.wi = nnx.Param(self.wi_distribution.sample(self.rngs))
        if self.bi is None:
            self.bi = nnx.Param(self.bi_distribution.sample(self.rngs))
        self.wi = jax.lax.stop_gradient(self.wi)
        self.bi = jax.lax.stop_gradient(self.bi)

        # Candidate gate
        if self.wc is None:
            self.wc = nnx.Param(self.wc_distribution.sample(self.rngs))
        if self.bc is None:
            self.bc = nnx.Param(self.bc_distribution.sample(self.rngs))
        self.wc = jax.lax.stop_gradient(self.wc)
        self.bc = jax.lax.stop_gradient(self.bc)

        # Output gate
        if self.wo is None:
            self.wo = nnx.Param(self.wo_distribution.sample(self.rngs))
        if self.bo is None:
            self.bo = nnx.Param(self.bo_distribution.sample(self.rngs))
        self.wo = jax.lax.stop_gradient(self.wo)
        self.bo = jax.lax.stop_gradient(self.bo)

        # Final output layer
        if self.wv is None:
            self.wv = nnx.Param(self.wv_distribution.sample(self.rngs))
        if self.bv is None:
            self.bv = nnx.Param(self.bv_distribution.sample(self.rngs))
        self.wv = jax.lax.stop_gradient(self.wv)
        self.bv = jax.lax.stop_gradient(self.bv)

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs for each pair of weights and bias
        log_probs_f = self.wf_distribution.log_prob(
            jnp.asarray(self.wf)
        ) + self.bf_distribution.log_prob(jnp.asarray(self.bf))
        log_probs_i = self.wi_distribution.log_prob(
            jnp.asarray(self.wi)
        ) + self.bi_distribution.log_prob(jnp.asarray(self.bi))
        log_probs_c = self.wc_distribution.log_prob(
            jnp.asarray(self.wc)
        ) + self.bc_distribution.log_prob(jnp.asarray(self.bc))
        log_probs_o = self.wo_distribution.log_prob(
            jnp.asarray(self.wo)
        ) + self.bo_distribution.log_prob(jnp.asarray(self.bo))
        log_probs_v = self.wv_distribution.log_prob(
            jnp.asarray(self.wv)
        ) + self.bv_distribution.log_prob(jnp.asarray(self.bv))

        # Compute the total loss
        log_probs = log_probs_f + log_probs_i + log_probs_c + log_probs_o + log_probs_v

        # Compute number of parameters
        num_params = (
            self.wf_distribution.num_params
            + self.bf_distribution.num_params
            + self.wi_distribution.num_params
            + self.bi_distribution.num_params
            + self.wc_distribution.num_params
            + self.bc_distribution.num_params
            + self.wo_distribution.num_params
            + self.bo_distribution.num_params
            + self.wv_distribution.num_params
            + self.bv_distribution.num_params
        )

        return log_probs, num_params

    def __call__(
        self,
        inputs: jax.Array,
        init_states: Optional[tuple[jax.Array, jax.Array]] = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Performs a forward pass through the Bayesian LSTM layer.
        If the layer is not frozen, it samples weights and bias
        from their respective distributions.

        Args:
            inputs: Input tensor with token indices. Shape: [batch, seq_len, 1]
            init_states: Optional initial hidden and cell states

        Returns:
            Tuple of (output, (hidden_state, cell_state))
        """

        # Sample weights if not frozen
        if not self.frozen:
            self.wf = nnx.Param(self.wf_distribution.sample(self.rngs))
            self.bf = nnx.Param(self.bf_distribution.sample(self.rngs))
            self.wi = nnx.Param(self.wi_distribution.sample(self.rngs))
            self.bi = nnx.Param(self.bi_distribution.sample(self.rngs))
            self.wc = nnx.Param(self.wc_distribution.sample(self.rngs))
            self.bc = nnx.Param(self.bc_distribution.sample(self.rngs))
            self.wo = nnx.Param(self.wo_distribution.sample(self.rngs))
            self.bo = nnx.Param(self.bo_distribution.sample(self.rngs))
            self.wv = nnx.Param(self.wv_distribution.sample(self.rngs))
            self.bv = nnx.Param(self.bv_distribution.sample(self.rngs))
        else:
            if any(w is None for w in [self.wf, self.wi, self.wc, self.wo, self.wv]):
                self.wf = nnx.Param(self.wf_distribution.sample(self.rngs))
                self.bf = nnx.Param(self.bf_distribution.sample(self.rngs))
                self.wi = nnx.Param(self.wi_distribution.sample(self.rngs))
                self.bi = nnx.Param(self.bi_distribution.sample(self.rngs))
                self.wc = nnx.Param(self.wc_distribution.sample(self.rngs))
                self.bc = nnx.Param(self.bc_distribution.sample(self.rngs))
                self.wo = nnx.Param(self.wo_distribution.sample(self.rngs))
                self.bo = nnx.Param(self.bo_distribution.sample(self.rngs))
                self.wv = nnx.Param(self.wv_distribution.sample(self.rngs))
                self.bv = nnx.Param(self.bv_distribution.sample(self.rngs))

        # Apply embedding layer to input indices
        inputs = jnp.squeeze(inputs, axis=-1)
        inputs = self.embedding(inputs)
        batch_size = jnp.shape(inputs)[0]
        seq_len = jnp.shape(inputs)[1]

        # Initialize h_t and c_t if init_states is None
        if init_states is None:
            h_t = jnp.zeros([batch_size, self.hidden_size])
            c_t = jnp.zeros([batch_size, self.hidden_size])
        else:
            h_t, c_t = init_states[0], init_states[1]

        # Process sequence
        for t in range(seq_len):
            # Shape: (batch_size, embedding_dim)
            x_t = inputs[:, t, :]

            # Concatenate input and hidden state
            # Shape: (batch_size, embedding_dim + hidden_size)
            z_t = jnp.concat([x_t, h_t], axis=1)

            # Forget gate
            ft = nnx.sigmoid(z_t @ self.wf.T + self.bf)

            # Input gate
            it = nnx.sigmoid(z_t @ self.wi.T + self.bi)

            # Candidate cell state
            can = nnx.tanh(z_t @ self.wc.T + self.bc)

            # Output gate
            ot = nnx.sigmoid(z_t @ self.wo.T + self.bo)

            # Update cell state
            c_t = c_t * ft + can * it

            # Update hidden state
            h_t = ot * nnx.tanh(c_t)

        # Compute final output
        y_t = h_t @ self.wv.T + self.bv

        return y_t, (h_t, c_t)
