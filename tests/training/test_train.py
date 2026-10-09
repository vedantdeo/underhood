"""Specification for training/train.py: the split, the batches, the loss estimate, then the run."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import underhood.training.train as train_module
from tests.conftest import GptFactory, TinyDims
from underhood import config, data
from underhood.training.train import estimate_loss, get_batch, split_data, train

CPU = torch.device("cpu")


@pytest.fixture
def stream(dims: TinyDims) -> list[int]:
    """A repetitive corpus: short enough to be fast, patterned enough to be learnable."""
    return [i % dims.vocab_size for i in range(1024)]


@pytest.mark.parametrize(
    ("val_fraction", "val_length"),
    [
        pytest.param(0.1, 100, id="the default tenth"),
        pytest.param(0.5, 500, id="an even split"),
        pytest.param(0.0, 0, id="no validation set at all"),
    ],
)
def test_split_data_holds_out_the_tail(val_fraction: float, val_length: int) -> None:
    ids = list(range(1000))
    train_data, val_data = split_data(ids, val_fraction)
    assert len(val_data) == val_length
    assert len(train_data) == 1000 - val_length
    assert train_data.dtype == torch.long
    assert val_data.tolist() == ids[1000 - val_length :]


@pytest.mark.parametrize(
    ("batch_size", "block_size"),
    [
        pytest.param(1, 1, id="one window of one token"),
        pytest.param(4, 8, id="a small batch"),
        pytest.param(32, 16, id="a training-sized batch"),
    ],
)
def test_get_batch_shapes(stream: list[int], batch_size: int, block_size: int) -> None:
    data = torch.tensor(stream, dtype=torch.long)
    x, y = get_batch(data, block_size, batch_size, CPU)
    assert x.shape == y.shape == (batch_size, block_size)
    assert x.dtype == torch.long


def test_get_batch_targets_are_the_inputs_shifted_one_on() -> None:
    data = torch.arange(500, dtype=torch.long)  # every value names its own position
    x, y = get_batch(data, block_size=8, batch_size=16, device=CPU)
    assert torch.equal(y, x + 1), "each target must be the token that follows its input"


def test_get_batch_is_reproducible_given_a_generator(stream: list[int]) -> None:
    data = torch.tensor(stream, dtype=torch.long)
    first = get_batch(data, 8, 4, CPU, generator=torch.Generator().manual_seed(3))
    again = get_batch(data, 8, 4, CPU, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first[0], again[0])
    assert torch.equal(first[1], again[1])


def test_get_batch_never_runs_off_the_end() -> None:
    data = torch.arange(20, dtype=torch.long)
    for _ in range(50):
        x, y = get_batch(data, block_size=8, batch_size=8, device=CPU)
        assert int(y.max().item()) <= 19, "a target window ran past the end of the data"
        assert int(x.min().item()) >= 0


def test_estimate_loss_reports_every_split(make_gpt: GptFactory, stream: list[int]) -> None:
    model = make_gpt()
    splits = {"train": torch.tensor(stream[:800]), "val": torch.tensor(stream[800:])}
    losses = estimate_loss(model, splits, CPU, batch_size=4, eval_iters=3)
    assert set(losses) == {"train", "val"}
    assert all(loss > 0 for loss in losses.values())


def test_estimate_loss_leaves_the_model_in_training_mode(
    make_gpt: GptFactory, stream: list[int]
) -> None:
    model = make_gpt().train()
    splits = {"train": torch.tensor(stream)}
    estimate_loss(model, splits, CPU, batch_size=4, eval_iters=2)
    assert model.training, "estimate_loss must put the model back the way it found it"


def test_estimate_loss_leaves_no_gradients_behind(make_gpt: GptFactory, stream: list[int]) -> None:
    model = make_gpt()
    estimate_loss(model, {"train": torch.tensor(stream)}, CPU, batch_size=4, eval_iters=2)
    assert all(p.grad is None for p in model.parameters()), "evaluation must not build a graph"


def test_train_returns_a_curve_that_starts_and_ends(
    make_gpt: GptFactory, stream: list[int]
) -> None:
    model = make_gpt()
    train_data, val_data = split_data(stream)
    history = train(
        model,
        train_data,
        val_data,
        CPU,
        max_iters=20,
        batch_size=4,
        eval_interval=10,
        eval_iters=2,
    )
    assert [row.iteration for row in history][0] == 0
    assert [row.iteration for row in history][-1] == 20


def test_train_drives_the_loss_down_on_a_learnable_corpus(
    make_gpt: GptFactory, stream: list[int]
) -> None:
    model = make_gpt()
    train_data, val_data = split_data(stream)
    history = train(
        model,
        train_data,
        val_data,
        CPU,
        max_iters=200,
        batch_size=16,
        learning_rate=1e-2,
        eval_interval=100,
        eval_iters=5,
    )
    first, last = history[0].train_loss, history[-1].train_loss
    assert last < first - 0.5, f"loss went {first:.3f} -> {last:.3f}; the loop is not learning"


def test_the_encoded_corpus_is_cached_and_rebuilt_when_the_vocab_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = tmp_path / "shakespeare.txt"
    text.write_text("to be or not to be, that is the question\n" * 20, encoding="utf-8")
    monkeypatch.setattr(data, "tiny_shakespeare", lambda: text)
    monkeypatch.setattr(train_module, "ENCODED", tmp_path / "bpe.pt")
    monkeypatch.setattr(config, "VOCAB_SIZE", 262)
    trained: list[int] = []
    train_bpe = train_module.BPETokenizer.train

    def counting(self: train_module.BPETokenizer, corpus: str, vocab_size: int) -> None:
        trained.append(vocab_size)
        train_bpe(self, corpus, vocab_size)

    monkeypatch.setattr(train_module.BPETokenizer, "train", counting)

    tokenizer, ids = train_module._encoded_corpus()
    again, cached_ids = train_module._encoded_corpus()
    monkeypatch.setattr(config, "VOCAB_SIZE", 264)
    train_module._encoded_corpus()

    assert tokenizer.decode(ids) == text.read_text(encoding="utf-8")
    assert (cached_ids, again.merges) == (ids, tokenizer.merges), "the second call read the cache"
    assert trained == [262, 264], "trained once, then again only for a new vocabulary size"
