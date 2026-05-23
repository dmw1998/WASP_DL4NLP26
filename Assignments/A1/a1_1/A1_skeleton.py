
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # Collect texts iterator
    if isinstance(train_file, str) and os.path.exists(train_file):
        def texts_iter():
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    else:
        # assume train_file is iterable of strings
        texts_iter = lambda: iter(train_file)

    counter = Counter()
    for t in texts_iter():
        toks = tokenize_fun(t)
        counter.update(toks)

    # Reserve slots for special tokens at the front
    specials = [pad_token, unk_token, bos_token, eos_token]
    most_common = [w for w, _ in counter.most_common()]
    if max_voc_size is not None:
        # subtract special tokens
        max_main = max(0, max_voc_size - len(specials))
        most_common = most_common[:max_main]

    vocab = specials + most_common
    str_to_int = {s: i for i, s in enumerate(vocab)}

    return A1Tokenizer(str_to_int=str_to_int, model_max_length=model_max_length,
                       pad_token=pad_token, unk_token=unk_token,
                       bos_token=bos_token, eos_token=eos_token)

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, model_max_length=None,
                 pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
        self.str_to_int = dict(str_to_int)
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}
        self.model_max_length = model_max_length

        # token ids
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.pad_token_id = self.str_to_int.get(pad_token, 0)
        self.unk_token_id = self.str_to_int.get(unk_token, 1)
        self.bos_token_id = self.str_to_int.get(bos_token, 2)
        self.eos_token_id = self.str_to_int.get(eos_token, 3)

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        if isinstance(texts, str):
            texts = [texts]

        sequences = []
        for t in texts:
            toks = lowercase_tokenizer(t) if isinstance(t, str) else list(t)
            ids = [self.bos_token_id] if self.bos_token_id is not None else []
            for w in toks:
                ids.append(self.str_to_int.get(w, self.unk_token_id))
            ids.append(self.eos_token_id)
            if truncation and self.model_max_length is not None:
                ids = ids[: self.model_max_length]
            sequences.append(ids)

        # padding
        if padding:
            maxlen = max(len(s) for s in sequences) if sequences else 0
            padded = [s + [self.pad_token_id] * (maxlen - len(s)) for s in sequences]
        else:
            padded = sequences

        if return_tensors == 'pt':
            import torch
            input_ids = torch.tensor(padded, dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            out = BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})
        else:
            out = BatchEncoding({'input_ids': padded, 'attention_mask': [[1 if tok != self.pad_token_id else 0 for tok in seq] for seq in padded]})

        return out

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        # Accept alternative names for compatibility
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size if embedding_size is not None else kwargs.get('embedding_dim')
        self.hidden_size = hidden_size if hidden_size is not None else kwargs.get('hidden_dim')

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        emb_size = config.embedding_size
        hid_size = config.hidden_size
        vocab_size = config.vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hid_size, batch_first=True)
        self.unembedding = nn.Linear(hid_size, vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        # input_ids: (B, T)
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        # rnn_out: (B, T, H)
        logits = self.unembedding(rnn_out)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        

        # optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # collate_fn: turn batch examples into tokenized tensors
        def collate_fn(batch):
            texts = [ex['text'] for ex in batch]
            enc = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
            input_ids = enc['input_ids']
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {'input_ids': input_ids, 'labels': labels}

        train_loader = DataLoader(self.train_dataset, batch_size=getattr(args, 'per_device_train_batch_size', 8), shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.eval_dataset, batch_size=getattr(args, 'per_device_eval_batch_size', 32), shuffle=False, collate_fn=collate_fn)

        for epoch in range(getattr(args, 'num_train_epochs', 1)):
            self.model.train()
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                out = self.model(input_ids, labels=labels)
                loss = out.loss if hasattr(out, 'loss') else out['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (i + 1) % max(1, getattr(args, 'log_every', 100)) == 0:
                    avg = running_loss / getattr(args, 'log_every', 100)
                    print(f'Epoch {epoch+1} step {i+1} avg_loss {avg:.4f}')
                    running_loss = 0.0

            # evaluation after epoch
            # evaluation after epoch: weight by number of non-padding tokens (most precise)
            self.model.eval()
            total_tokens = 0
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    out = self.model(input_ids, labels=labels)
                    loss = out.loss if hasattr(out, 'loss') else out['loss']
                    # count valid (non -100) tokens in labels
                    num_tokens = int((labels != -100).sum().item())
                    if num_tokens > 0:
                        val_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
            if total_tokens > 0:
                val_loss = val_loss / total_tokens
                import math
                ppl = math.exp(val_loss) if val_loss < 100 else float('inf')
                print(f'Validation loss after epoch {epoch+1}: {val_loss:.4f}  ppl: {ppl:.2f}')

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    
