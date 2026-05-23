"""
Part 1: Tokenization — implementation that fills in the skeleton.

Provides:
    - lowercase_tokenizer:  the default word-splitting function (NLTK)
    - build_tokenizer:      build vocab from a training file, return A1Tokenizer
    - A1Tokenizer:          HuggingFace-like tokenizer (__call__ / __len__ / save / from_file)
"""

import pickle
from collections import Counter

import torch
import nltk
from transformers import BatchEncoding


def lowercase_tokenizer(text):
    """Default word-splitting function: NLTK word_tokenize + lowercase."""
    return [t.lower() for t in nltk.word_tokenize(text)]


def build_tokenizer(
    train_file,
    tokenize_fun=lowercase_tokenizer,
    max_voc_size=None,
    model_max_length=None,
    pad_token='<PAD>',
    unk_token='<UNK>',
    bos_token='<BOS>',
    eos_token='<EOS>',
):
    """Build a tokenizer from a training file.

    Reads `train_file` line by line (each nonempty line = one paragraph),
    counts word frequencies, and creates a vocabulary mapping. The 4
    special tokens occupy the first 4 ids; the remaining slots are
    filled with the most frequent words.

    Returns:
        An A1Tokenizer instance.
    """
    counter = Counter()
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counter.update(tokenize_fun(line))

    # Reserve first 4 ids for specials. Order matters: pad must be at a
    # stable id so we can save its id as pad_token_id.
    specials = [pad_token, unk_token, bos_token, eos_token]
    str_to_int = {tok: i for i, tok in enumerate(specials)}

    if max_voc_size is None:
        candidates = counter.most_common()
    else:
        candidates = counter.most_common(max_voc_size - len(specials))

    for word, _ in candidates:
        if word in str_to_int:        # avoid collision with a special token
            continue
        str_to_int[word] = len(str_to_int)

    return A1Tokenizer(
        str_to_int=str_to_int,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        tokenize_fun=tokenize_fun,
        model_max_length=model_max_length,
    )


class A1Tokenizer:
    """A minimal HuggingFace-like tokenizer."""

    def __init__(
        self,
        str_to_int,
        pad_token='<PAD>',
        unk_token='<UNK>',
        bos_token='<BOS>',
        eos_token='<EOS>',
        tokenize_fun=lowercase_tokenizer,
        model_max_length=None,
    ):
        self.str_to_int = str_to_int
        self.int_to_str = {i: w for w, i in str_to_int.items()}

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Required attributes the skeleton calls out explicitly.
        self.pad_token_id = str_to_int[pad_token]
        self.model_max_length = model_max_length

        # Other convenient ids.
        self.unk_token_id = str_to_int[unk_token]
        self.bos_token_id = str_to_int[bos_token]
        self.eos_token_id = str_to_int[eos_token]

        self.tokenize_fun = tokenize_fun

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_one(self, text):
        """Encode one string: tokenize, wrap with BOS/EOS, map to ids."""
        ids = [self.bos_token_id]
        for tok in self.tokenize_fun(text):
            ids.append(self.str_to_int.get(tok, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize one string or a list of strings.

        Returns a BatchEncoding with 'input_ids' (and 'attention_mask').
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')

        # Accept both a single string and a list of strings.
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self._encode_one(t) for t in texts]

        if truncation and self.model_max_length is not None:
            encoded = [seq[: self.model_max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            attention_mask = [
                [1] * len(seq) + [0] * (max_len - len(seq)) for seq in encoded
            ]
            input_ids = [
                seq + [self.pad_token_id] * (max_len - len(seq))
                for seq in encoded
            ]
        else:
            input_ids = encoded
            attention_mask = [[1] * len(seq) for seq in encoded]

        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return BatchEncoding(
            {'input_ids': input_ids, 'attention_mask': attention_mask}
        )

    def __len__(self):
        return len(self.str_to_int)

    # ------------------------------------------------------------------
    # Decoding (convenience, used in Part 5)
    # ------------------------------------------------------------------
    def decode(self, ids, skip_special=True):
        """Map a list (or 1D tensor) of ids back to a list of word strings."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        specials = (
            {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
            if skip_special else set()
        )
        out = []
        for i in ids:
            tok = self.int_to_str.get(int(i), self.unk_token)
            if tok in specials:
                continue
            out.append(tok)
        return out

    # ------------------------------------------------------------------
    # Persistence — skeleton provides these, kept identical
    # ------------------------------------------------------------------
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)