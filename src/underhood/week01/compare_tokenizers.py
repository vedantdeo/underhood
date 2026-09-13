"""Your BPE against tiktoken on the same text. Week 1, Track B.

    uv run underhood-bpe-compare

Trains your tokenizer on the first part of tiny Shakespeare, then reports bytes per token for
yours and for two of OpenAI's tokenizers on held-out text. Tiktoken's vocabularies are ~100x
larger and trained on far more data; the point is to see what vocabulary size buys.
"""

from __future__ import annotations

import tiktoken

from underhood.data import tiny_shakespeare
from underhood.week01.bpe import BPETokenizer


def main(vocab_size: int = 512, train_chars: int = 200_000) -> None:
    text = tiny_shakespeare().read_text(encoding="utf-8")
    train_text, held_out = text[:train_chars], text[train_chars : train_chars + 50_000]

    mine = BPETokenizer()
    mine.train(train_text, vocab_size)
    held_bytes = len(held_out.encode("utf-8"))

    rows: list[tuple[str, int, int]] = [
        (f"yours (vocab {vocab_size})", vocab_size, len(mine.encode(held_out))),
    ]
    for name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        rows.append((f"tiktoken {name}", enc.n_vocab, len(enc.encode(held_out))))

    print(f"held-out text: {held_bytes:,} bytes\n")
    print(f"{'tokenizer':<28}{'vocab':>10}{'tokens':>10}{'bytes/token':>14}")
    for name, vocab, tokens in rows:
        print(f"{name:<28}{vocab:>10,}{tokens:>10,}{held_bytes / tokens:>14.2f}")


if __name__ == "__main__":
    main()
