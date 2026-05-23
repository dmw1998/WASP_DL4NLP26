"""
Part 3: Defining the language model neural network.

Implements an RNN-based autoregressive language model following the
HuggingFace Transformers conventions (PretrainedConfig + PreTrainedModel).
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class A1RNNModelConfig(PretrainedConfig):
    """Configuration object holding all hyperparameters of the model."""

    model_type = "a1_rnn"

    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        dropout=0.0,
        pad_token_id=3,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout


class A1RNNModel(PreTrainedModel):
    """RNN-based autoregressive language model.

    Architecture:
        token ids -> Embedding -> LSTM -> Linear -> logits over vocabulary.

    The forward pass mirrors the HuggingFace API: it accepts `input_ids`
    and an optional `labels` tensor. When `labels` is given, the loss
    is computed using the standard "shift-by-one" rule.
    """

    config_class = A1RNNModelConfig

    def __init__(self, config: A1RNNModelConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        # We use LSTM; GRU works too. batch_first=True so input is (B, N, E).
        self.rnn = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output (unembedding) layer: hidden state -> vocabulary logits.
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

        # Standard HuggingFace weight initialization.
        self.post_init()

    def _init_weights(self, module):
        """Weight init used by PreTrainedModel.post_init()."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass.

        Args:
            input_ids:    LongTensor (B, N).
            labels:       LongTensor (B, N) or None. Positions to ignore in
                          the loss should be set to -100 (the
                          CrossEntropyLoss default ignore_index).
            attention_mask: optional, unused here but accepted for API
                          compatibility.

        Returns:
            dict with 'logits' (B, N, V) and optionally 'loss'.
        """
        embedded = self.embedding(input_ids)          # (B, N, E)
        rnn_out, _ = self.rnn(embedded)               # (B, N, H)
        logits = self.output(rnn_out)                 # (B, N, V)

        loss = None
        if labels is not None:
            # Shift: predict token i+1 from token i.
            #   - drop the last logit (nothing follows it)
            #   - drop the first label (nothing precedes it)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}