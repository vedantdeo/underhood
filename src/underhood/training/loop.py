"""The training loop: batches out of a token stream, a loss estimate, and the run itself.

The tests in tests/training/test_loop.py are the specification, in this order: split_data,
get_batch, estimate_loss, then train. main() is already written — it is the plumbing that turns
your loop into a loss curve, a checkpoint, and the ms/iter number that settles config.py's guesses.

uv run underhood-train
"""

from __future__ import annotations

import time
from typing import NamedTuple

import torch
from torch import Tensor

from underhood import config, data
from underhood.device import pick_device
from underhood.model.gpt import GPT
from underhood.tokenizer.bpe import BPETokenizer

CHECKPOINT = data.DATA_DIR / "gpt.pt"
ENCODED = data.DATA_DIR / "tinyshakespeare-bpe.pt"


class Snapshot(NamedTuple):
    """One row of the loss curve."""

    iteration: int
    train_loss: float
    val_loss: float


def split_data(ids: list[int], val_fraction: float = config.VAL_FRACTION) -> tuple[Tensor, Tensor]:
    """Cut a token stream into (train, val) int64 tensors, the last val_fraction held out.

    Held out from the end rather than sampled at random: neighbouring windows of text overlap, so
    a random split leaks the training set into validation almost everywhere.
    """
    raise NotImplementedError


def get_batch(
    data_: Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """batch_size random windows: x is (batch_size, block_size), y the same window shifted one on.

    Every start offset must leave room for both, so offsets come from [0, len(data_) - block_size).
    Draw them with `generator` when one is given, so a test can reproduce a batch.
    """
    raise NotImplementedError


def estimate_loss(
    model: GPT,
    splits: dict[str, Tensor],
    device: torch.device,
    batch_size: int = config.BATCH_SIZE,
    eval_iters: int = config.EVAL_ITERS,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Mean loss over eval_iters batches of each split, keyed by the same names as `splits`.

    Runs with gradients off and the model in eval(), and leaves it back in train() on the way out.
    A single batch's loss is too noisy to read a curve from; this is the number worth printing.
    """
    raise NotImplementedError


def train(
    model: GPT,
    train_data: Tensor,
    val_data: Tensor,
    device: torch.device,
    max_iters: int = config.MAX_ITERS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    eval_interval: int = config.EVAL_INTERVAL,
    eval_iters: int = config.EVAL_ITERS,
    generator: torch.Generator | None = None,
) -> list[Snapshot]:
    """Train with AdamW for max_iters, returning the loss curve as one Snapshot per evaluation.

    Evaluate at iteration 0, every eval_interval after that, and once more at the end, so the
    returned curve always has a first and last row to compare.
    """
    raise NotImplementedError


def _encoded_corpus() -> tuple[BPETokenizer, list[int]]:
    """Train the BPE on tiny Shakespeare and encode it, caching the result under data/.

    The week-1 BPE is the naive one, so this is minutes of Python the first time and instant after.
    """
    text = data.tiny_shakespeare().read_text(encoding="utf-8")
    if ENCODED.exists():
        cached = torch.load(ENCODED, weights_only=False)
        if cached["vocab_size"] == config.VOCAB_SIZE:
            tokenizer = BPETokenizer()
            tokenizer.merges = cached["merges"]
            tokenizer.vocab = cached["vocab"]
            return tokenizer, cached["ids"]

    started = time.perf_counter()
    tokenizer = BPETokenizer()
    # the tokenizer never sees the validation text; split_data cuts the ids at the same fraction
    tokenizer.train(text[: int(len(text) * (1 - config.VAL_FRACTION))], config.VOCAB_SIZE)
    ids = tokenizer.encode(text)
    print(
        f"tokenized {len(text):,} chars into {len(ids):,} ids "
        f"({len(text) / len(ids):.2f} chars/token) in {time.perf_counter() - started:.1f}s"
    )
    torch.save(
        {
            "ids": ids,
            "merges": tokenizer.merges,
            "vocab": tokenizer.vocab,
            "vocab_size": config.VOCAB_SIZE,
        },
        ENCODED,
    )
    return tokenizer, ids


def main() -> None:
    torch.manual_seed(config.SEED)
    device = pick_device()
    tokenizer, ids = _encoded_corpus()
    train_data, val_data = split_data(ids)

    model = GPT(vocab_size=len(tokenizer.vocab)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  params={params:,}  "
        f"train={len(train_data):,} val={len(val_data):,} tokens"
    )

    started = time.perf_counter()
    history = train(model, train_data, val_data, device)
    elapsed = time.perf_counter() - started

    print(f"\n{'iter':>7}  {'train':>8}  {'val':>8}")
    for row in history:
        print(f"{row.iteration:>7}  {row.train_loss:>8.4f}  {row.val_loss:>8.4f}")

    seen = config.MAX_ITERS * config.BATCH_SIZE * config.BLOCK_SIZE
    print(
        f"\n{config.MAX_ITERS} iters in {elapsed:.1f}s  "
        f"({elapsed / config.MAX_ITERS * 1000:.1f} ms/iter, {seen / elapsed:,.0f} tokens/s)"
    )

    torch.save(
        {
            "block_size": config.BLOCK_SIZE,
            "merges": tokenizer.merges,
            "model": model.state_dict(),
            "vocab": tokenizer.vocab,
            "vocab_size": len(tokenizer.vocab),
        },
        CHECKPOINT,
    )
    print(f"checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
