"""
Part 1: Tokenization.

Implements vocabulary building and a HuggingFace-like tokenizer
(A1Tokenizer) for the assignment.
"""

import json
from collections import Counter

import torch
from nltk.tokenize import word_tokenize


# Special symbols. Using UPPERCASE so they cannot collide with real lowercase
# tokens (we lowercase all real tokens before adding them to the vocab).
BOS_TOKEN = "BEGINNING"
EOS_TOKEN = "END"
UNK_TOKEN = "UNKNOWN"
PAD_TOKEN = "PAD"
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]


def build_vocab(texts, max_voc_size):
    """Build a string -> int vocabulary from an iterable of raw texts.

    The 4 special tokens are guaranteed to be in the vocabulary and occupy
    the first 4 integer ids. Remaining slots are filled with the most
    frequent words from the corpus, until `max_voc_size` is reached.
    """
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower())
        counter.update(tokens)

    str_to_int = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

    # Reserve the first 4 ids for special tokens.
    remaining = max_voc_size - len(SPECIAL_TOKENS)
    for word, _ in counter.most_common(remaining):
        # Defensive: skip if a real token happens to match a special token.
        if word in str_to_int:
            continue
        str_to_int[word] = len(str_to_int)

    int_to_str = {i: w for w, i in str_to_int.items()}
    return str_to_int, int_to_str


class A1Tokenizer:
    """Functionally similar to a HuggingFace tokenizer.

    Use `build_tokenizer(...)` to construct from raw texts, or
    `A1Tokenizer.from_file(...)` to load a saved one.
    """

    def __init__(self, str_to_int):
        self.str_to_int = str_to_int
        self.int_to_str = {i: w for w, i in str_to_int.items()}

        # Convenience attributes for special token ids.
        self.bos_id = str_to_int[BOS_TOKEN]
        self.eos_id = str_to_int[EOS_TOKEN]
        self.unk_id = str_to_int[UNK_TOKEN]
        self.pad_id = str_to_int[PAD_TOKEN]

    def __len__(self):
        return len(self.str_to_int)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_one(self, text):
        """Encode a single string: lowercase, tokenize, wrap with BOS/EOS,
        and look up ids. Unknown words map to UNK."""
        tokens = word_tokenize(text.lower())
        ids = [self.bos_id]
        for tok in tokens:
            ids.append(self.str_to_int.get(tok, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
    ):
        """Tokenize a list of strings.

        Args:
            texts: a single string or a list of strings.
            return_tensors: if 'pt', returns torch tensors.
            padding: if True, pads sequences in the batch to the same length
                using the PAD id (right-padding).
            truncation: if True and `max_length` is given, truncates each
                sequence to `max_length` tokens.
            max_length: max sequence length when truncating.

        Returns:
            dict with keys 'input_ids' and 'attention_mask'.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self._encode_one(t) for t in texts]

        if truncation and max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            attention_mask = [
                [1] * len(seq) + [0] * (max_len - len(seq)) for seq in encoded
            ]
            input_ids = [
                seq + [self.pad_id] * (max_len - len(seq)) for seq in encoded
            ]
        else:
            input_ids = encoded
            attention_mask = [[1] * len(seq) for seq in encoded]

        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------
    def decode(self, ids, skip_special=True):
        """Map a list (or 1D tensor) of ids back to a list of strings."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out = []
        specials = set(SPECIAL_TOKENS) if skip_special else set()
        for i in ids:
            tok = self.int_to_str.get(int(i), UNK_TOKEN)
            if tok in specials:
                continue
            out.append(tok)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.str_to_int, f, ensure_ascii=False)

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            str_to_int = json.load(f)
        # JSON only allows string keys, so the loaded dict is already correct.
        return cls(str_to_int)


def build_tokenizer(texts, max_voc_size):
    """Convenience: build the vocabulary from raw texts and wrap it in
    an A1Tokenizer."""
    str_to_int, _ = build_vocab(texts, max_voc_size)
    return A1Tokenizer(str_to_int)