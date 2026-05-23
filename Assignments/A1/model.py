"""
Part 3: Defining the language model neural network.

Fills in the skeleton: A1RNNModelConfig + A1RNNModel.
The model is an RNN-based autoregressive language model that returns
HuggingFace's CausalLMOutput.
"""

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class A1RNNModelConfig(PretrainedConfig):
    """Configuration object storing the model hyperparameters.

    Field names (vocab_size / embedding_size / hidden_size) match the
    skeleton exactly.
    """

    model_type = 'a1_rnn'

    def __init__(
        self,
        vocab_size=10000,
        embedding_size=128,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout


class A1RNNModel(PreTrainedModel):
    """RNN-based autoregressive language model.

    Architecture: token ids -> Embedding -> LSTM -> Linear (unembedding).
    """

    config_class = A1RNNModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)

        # LSTM with batch_first=True so inputs are (B, N, E).
        # We avoid plain nn.RNN as suggested by the assignment.
        self.rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Unembedding (output) layer: hidden state -> vocabulary logits.
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

        # Loss with ignore_index=-100 (HuggingFace convention for "ignore").
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        """Forward pass.

        Args:
            input_ids: LongTensor (B, N).
            labels:    LongTensor (B, N) or None. Positions to ignore
                       in the loss should be -100.

        Returns:
            CausalLMOutput with `logits` (B, N, V) and optional `loss`.
        """
        embedded = self.embedding(input_ids)          # (B, N, E)
        rnn_out, _ = self.rnn(embedded)               # (B, N, H)
        logits = self.unembedding(rnn_out)            # (B, N, V)

        loss = None
        if labels is not None:
            # Shift by one: predict token i+1 from token i.
            # We observe nothing after the last position -> drop last logit.
            # We observe nothing before the first position -> drop first label.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # CrossEntropyLoss expects 2D logits (N, V) and 1D labels (N).
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutput(logits=logits, loss=loss)