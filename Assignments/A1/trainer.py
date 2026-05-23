"""
Part 4: Training the language model.

Fills in the skeleton's A1Trainer.train() method.
"""

import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader


class A1Trainer:
    """Minimal trainer that mirrors HuggingFace's Trainer API."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert args.optim == 'adamw_torch'
        assert args.eval_strategy == 'epoch'

    def select_device(self):
        """Pick CPU / CUDA / MPS based on args and available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not getattr(self.args, 'no_cuda', False) and torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch, 'mps') and torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    # ------------------------------------------------------------------
    # Collation: a HuggingFace dataset row is {'text': ...}. We turn a
    # list of rows into a single (input_ids, labels) tensor pair.
    # ------------------------------------------------------------------
    def _collate(self, batch):
        # `batch` is a list of dict rows from the underlying dataset.
        texts = [row['text'] for row in batch]
        enc = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        )
        input_ids = enc['input_ids']

        # Labels = input_ids, with padding positions replaced by -100
        # so they're ignored by CrossEntropyLoss.
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

    # ------------------------------------------------------------------
    # Train / evaluate
    # ------------------------------------------------------------------
    def train(self):
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(args, 'weight_decay', 0.0),
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )
        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )

        log_every = getattr(args, 'logging_steps', 100) or 100

        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            t0 = time.time()
            running_loss = 0.0
            running_count = 0

            for input_ids, labels in train_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # Forward + loss (loss is computed inside the model).
                out = self.model(input_ids=input_ids, labels=labels)
                loss = out.loss

                # Backward + step.
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping stabilizes RNN training.
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                running_count += 1
                global_step += 1

                if global_step % log_every == 0:
                    avg = running_loss / running_count
                    print(
                        f'epoch {epoch+1} | step {global_step} | '
                        f'train_loss {avg:.4f} | train_ppl {math.exp(avg):.2f}'
                    )
                    running_loss = 0.0
                    running_count = 0

            # End-of-epoch evaluation (matches eval_strategy='epoch').
            val_loss, val_ppl = self.evaluate(val_loader=val_loader, device=device)
            elapsed = time.time() - t0
            print(
                f'epoch {epoch+1} DONE in {elapsed:.1f}s | '
                f'val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}'
            )

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    @torch.no_grad()
    def evaluate(self, val_loader=None, device=None):
        """Compute mean cross-entropy loss and perplexity on the validation set."""
        if device is None:
            device = self.select_device()
            self.model.to(device)
        if val_loader is None:
            val_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            out = self.model(input_ids=input_ids, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        # Perplexity = exp(mean cross-entropy). Natural log + exp is fine
        # as long as we're consistent.
        ppl = math.exp(avg_loss)
        return avg_loss, ppl