"""
This module contains the code for the bayesian LSTM.
"""

# Standard libraries
from typing import Any, Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf.gaussian import GaussianDistribution
from illia.nn.tf.base import BayesianModule
from illia.nn.tf.embedding import Embedding


@saving.register_keras_serializable(package="BayesianModule", name="LSTM")
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
        sparse: bool = False,
        **kwargs: Any,
    ) -> None:

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

        # Initialize weight distributions
        # Forget gate
        self.wf_distribution = GaussianDistribution(
            (self.embeddings_dim + self.hidden_size, self.hidden_size)
        )
        self.bf_distribution = GaussianDistribution((self.hidden_size,))

        # Input gate
        self.wi_distribution = GaussianDistribution(
            (self.embeddings_dim + self.hidden_size, self.hidden_size)
        )
        self.bi_distribution = GaussianDistribution((self.hidden_size,))

        # Candidate gate
        self.wc_distribution = GaussianDistribution(
            (self.embeddings_dim + self.hidden_size, self.hidden_size)
        )
        self.bc_distribution = GaussianDistribution((self.hidden_size,))

        # Output gate
        self.wo_distribution = GaussianDistribution(
            (self.embeddings_dim + self.hidden_size, self.hidden_size)
        )
        self.bo_distribution = GaussianDistribution((self.hidden_size,))

        # Final output layer
        self.wv_distribution = GaussianDistribution(
            (self.hidden_size, self.output_size)
        )
        self.bv_distribution = GaussianDistribution((self.output_size,))

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the Bayesian LSTM layer by creating all gate weights and biases.

        Args:
            input_shape: Input shape of the layer.
        """

        # Forget gate weights and bias
        self.wf = self.add_weight(
            name="forget_gate_weights",
            initializer=tf.constant_initializer(self.wf_distribution.sample().numpy()),
            shape=(self.embeddings_dim + self.hidden_size, self.hidden_size),
            trainable=False,
        )

        self.bf = self.add_weight(
            name="forget_gate_bias",
            initializer=tf.constant_initializer(self.bf_distribution.sample().numpy()),
            shape=(self.hidden_size,),
            trainable=False,
        )

        # Input gate weights and bias
        self.wi = self.add_weight(
            name="input_gate_weights",
            initializer=tf.constant_initializer(self.wi_distribution.sample().numpy()),
            shape=(self.embeddings_dim + self.hidden_size, self.hidden_size),
            trainable=False,
        )

        self.bi = self.add_weight(
            name="input_gate_bias",
            initializer=tf.constant_initializer(self.bi_distribution.sample().numpy()),
            shape=(self.hidden_size,),
            trainable=False,
        )

        # Candidate gate weights and bias
        self.wc = self.add_weight(
            name="candidate_gate_weights",
            initializer=tf.constant_initializer(self.wc_distribution.sample().numpy()),
            shape=(self.embeddings_dim + self.hidden_size, self.hidden_size),
            trainable=False,
        )

        self.bc = self.add_weight(
            name="candidate_gate_bias",
            initializer=tf.constant_initializer(self.bc_distribution.sample().numpy()),
            shape=(self.hidden_size,),
            trainable=False,
        )

        # Output gate weights and bias
        self.wo = self.add_weight(
            name="output_gate_weights",
            initializer=tf.constant_initializer(self.wo_distribution.sample().numpy()),
            shape=(self.embeddings_dim + self.hidden_size, self.hidden_size),
            trainable=False,
        )

        self.bo = self.add_weight(
            name="output_gate_bias",
            initializer=tf.constant_initializer(self.bo_distribution.sample().numpy()),
            shape=(self.hidden_size,),
            trainable=False,
        )

        # Final output layer weights and bias
        self.wv = self.add_weight(
            name="final_output_weights",
            initializer=tf.constant_initializer(self.wv_distribution.sample().numpy()),
            shape=(self.hidden_size, self.output_size),
            trainable=False,
        )

        self.bv = self.add_weight(
            name="final_output_bias",
            initializer=tf.constant_initializer(self.bv_distribution.sample().numpy()),
            shape=(self.output_size,),
            trainable=False,
        )

        # Call super-class build method
        super().build(input_shape)

    def get_config(self) -> dict:
        """
        Retrieves the configuration of the Conv2d layer.

        Returns:
            Dictionary containing layer configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "num_embeddings": self.num_embeddings,
            "embeddings_dim": self.embeddings_dim,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "padding_idx": self.padding_idx,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
            "scale_grad_by_freq": self.scale_grad_by_freq,
            "sparse": self.sparse,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
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
        self.wf = tf.stop_gradient(self.wf)
        self.bf = tf.stop_gradient(self.bf)

        # Input gate
        if self.wi is None:
            self.wi = self.wi_distribution.sample()
        if self.bi is None:
            self.bi = self.bi_distribution.sample()
        self.wi = tf.stop_gradient(self.wi)
        self.bi = tf.stop_gradient(self.bi)

        # Candidate gate
        if self.wc is None:
            self.wc = self.wc_distribution.sample()
        if self.bc is None:
            self.bc = self.bc_distribution.sample()
        self.wc = tf.stop_gradient(self.wc)
        self.bc = tf.stop_gradient(self.bc)

        # Output gate
        if self.wo is None:
            self.wo = self.wo_distribution.sample()
        if self.bo is None:
            self.bo = self.bo_distribution.sample()
        self.wo = tf.stop_gradient(self.wo)
        self.bo = tf.stop_gradient(self.bo)

        # Final output layer
        if self.wv is None:
            self.wv = self.wv_distribution.sample()
        if self.bv is None:
            self.bv = self.bv_distribution.sample()
        self.wv = tf.stop_gradient(self.wv)
        self.bv = tf.stop_gradient(self.bv)

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs for each pair of weights and bias
        log_probs_f = self.wf_distribution.log_prob(
            self.wf
        ) + self.bf_distribution.log_prob(self.bf)

        log_probs_i = self.wi_distribution.log_prob(
            self.wi
        ) + self.bi_distribution.log_prob(self.bi)

        log_probs_c = self.wc_distribution.log_prob(
            self.wc
        ) + self.bc_distribution.log_prob(self.bc)

        log_probs_o = self.wo_distribution.log_prob(
            self.wo
        ) + self.bo_distribution.log_prob(self.bo)

        log_probs_v = self.wv_distribution.log_prob(
            self.wv
        ) + self.bv_distribution.log_prob(self.bv)

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

    def call(
        self,
        inputs: tf.Tensor,
        init_states: Optional[tuple[tf.Tensor, tf.Tensor]] = None,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
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
        else:
            if any(w is None for w in [self.wf, self.wi, self.wc, self.wo, self.wv]):
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

        # Apply embedding layer to input indices
        inputs = tf.squeeze(inputs, axis=-1)
        inputs = self.embedding(inputs)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Initialize h_t and c_t if init_states is None
        if init_states is None:
            h_t = tf.zeros([batch_size, self.hidden_size], dtype=inputs.dtype)
            c_t = tf.zeros([batch_size, self.hidden_size], dtype=inputs.dtype)
        else:
            h_t, c_t = init_states[0], init_states[1]

        # Process sequence
        for t in range(seq_len):
            # Shape: (batch_size, embedding_dim)
            x_t = inputs[:, t, :]

            # Concatenate input and hidden state
            # Shape: (batch_size, embedding_dim + hidden_size)
            z_t = tf.concat([x_t, h_t], axis=1)

            # Forget gate
            ft = tf.sigmoid(tf.matmul(z_t, self.wf) + self.bf)

            # Input gate
            it = tf.sigmoid(tf.matmul(z_t, self.wi) + self.bi)

            # Candidate cell state
            can = tf.tanh(tf.matmul(z_t, self.wc) + self.bc)

            # Output gate
            ot = tf.sigmoid(tf.matmul(z_t, self.wo) + self.bo)

            # Update cell state
            c_t = c_t * ft + can * it

            # Update hidden state
            h_t = ot * tf.tanh(c_t)

        # Compute final output
        y_t = tf.matmul(h_t, self.wv) + self.bv

        return y_t, (h_t, c_t)
