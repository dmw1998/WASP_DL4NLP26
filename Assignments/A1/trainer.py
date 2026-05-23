"""
Part 4: Training the model.

Implements a minimal Trainer mimicking HuggingFace's `Trainer` API.
"""

import math
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader


@dataclass
class A1TrainingArguments:
    """Subset of HuggingFace's TrainingArguments that we actually need."""

    output_dir: str = "trainer_output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_seq_length: int = 128
    log_every: int = 100  # log training loss every N steps


class A1Trainer:
    """Minimal trainer for our language model."""

    def __init__(self, model, tokenizer, args: A1TrainingArguments,
                 train_dataset=None, eval_dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collate(self, batch):
        """Take a list of dataset items (each a dict with 'text') and turn
        it into an `input_ids`/`labels` tensor pair ready for the model."""
        texts = [item["text"] for item in batch]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_length,
        )
        input_ids = encoded["input_ids"]

        # Labels are identical to input_ids, except padding positions
        # are replaced with -100 so they don't contribute to the loss.
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_id] = -100
        return input_ids, labels

    def _make_loader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
        )

    # ------------------------------------------------------------------
    # Train / Evaluate
    # ------------------------------------------------------------------
    def train(self):
        train_loader = self._make_loader(
            self.train_dataset,
            self.args.per_device_train_batch_size,
            shuffle=True,
        )
        eval_loader = None
        if self.eval_dataset is not None:
            eval_loader = self._make_loader(
                self.eval_dataset,
                self.args.per_device_eval_batch_size,
                shuffle=False,
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        global_step = 0
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            running_loss = 0.0
            running_count = 0

            for input_ids, labels in train_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                out = self.model(input_ids=input_ids, labels=labels)
                loss = out["loss"]
                loss.backward()
                # Gradient clipping: standard trick to stabilize RNN training.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                running_count += 1
                global_step += 1

                if global_step % self.args.log_every == 0:
                    avg = running_loss / running_count
                    print(
                        f"epoch {epoch+1} | step {global_step} | "
                        f"train_loss {avg:.4f} | train_ppl {math.exp(avg):.2f}"
                    )
                    running_loss = 0.0
                    running_count = 0

            # Validation at end of each epoch.
            if eval_loader is not None:
                val_loss, val_ppl = self.evaluate(eval_loader)
                print(
                    f"epoch {epoch+1} DONE | val_loss {val_loss:.4f} | "
                    f"val_ppl {val_ppl:.2f}"
                )

        # Save the trained model in HuggingFace format.
        self.model.save_pretrained(self.args.output_dir)
        print(f"Model saved to {self.args.output_dir}")

    @torch.no_grad()
    def evaluate(self, eval_loader=None):
        """Compute average cross-entropy loss and perplexity on a dataset."""
        if eval_loader is None:
            eval_loader = self._make_loader(
                self.eval_dataset,
                self.args.per_device_eval_batch_size,
                shuffle=False,
            )
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        for input_ids, labels in eval_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            out = self.model(input_ids=input_ids, labels=labels)
            total_loss += out["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        # Perplexity = exp(mean cross-entropy). Base doesn't matter as long
        # as we're consistent; natural log + exp is the convenient choice.
        ppl = math.exp(avg_loss)
        return avg_loss, ppl