"""Byte-pair encoding from scratch. Week 1, Track B.

The tests in tests/test_bpe.py are the specification. Make them green one function at a time:
get_stats, then merge, then train, then encode, then decode.

Reference: Karpathy's "Let's build the GPT tokenizer" and minbpe. Do not read minbpe's code until
yours passes; then read it and compare. When done, run `uv run underhood-bpe-compare` to see how
your tokenizer's compression stacks up against tiktoken's on the same text.
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

Pair = tuple[int, int]


def get_stats(ids: list[int]) -> dict[Pair, int]:
    """Count how often each adjacent pair of ids occurs.

    get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    return Counter(pairwise(ids))


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Return a new list with every occurrence of `pair` replaced by `new_id`.

    merge([5, 6, 6, 7, 9, 1], (6, 7), 99) == [5, 6, 99, 9, 1]
    """
    new_ids: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """Byte-level BPE: start from the 256 byte values, learn merges by frequency."""

    def __init__(self) -> None:
        # (a, b) -> new token id, inserted in training order. Order matters for encode.
        self.merges: dict[Pair, int] = {}
        # token id -> the bytes it stands for. Ids 0..255 are the raw bytes.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        """Learn merges from `text` until `vocab` reaches `vocab_size`, most frequent pair first.

        Each merge takes the next free id and extends `vocab` with the concatenated bytes of its
        two parts. Calling train() again resumes from the current merges rather than starting over.
        """
        prev_vocab_size = len(self.vocab)
        if vocab_size < prev_vocab_size:
            raise ValueError(
                f"vocab_size {vocab_size} is smaller than current vocab size"
                f" {prev_vocab_size}. Contracting the vocab is not supported."
            )

        ids = self.encode(text)
        for next_id in range(prev_vocab_size, vocab_size):
            pair_counts = get_stats(ids)
            if not pair_counts:
                break
            most_frequent_pair = max(pair_counts, key=pair_counts.__getitem__)
            a, b = most_frequent_pair
            self.merges[most_frequent_pair] = next_id
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]
            ids = merge(ids, most_frequent_pair, next_id)

    def encode(self, text: str) -> list[int]:
        """Text -> UTF-8 bytes -> apply the merges in training order -> token ids."""
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            ids = merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids -> bytes -> text. Use errors="replace": a split multi-byte character must not
        crash the decoder."""
        bytes_data = b"".join(self.vocab[token_id] for token_id in ids)
        return bytes_data.decode("utf-8", errors="replace")
