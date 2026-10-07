"""Specification for inference/sampling.py: the three filters, then the loop that uses them."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import underhood.inference.sampling as sampling
from tests.conftest import GptFactory, TinyDims
from underhood.inference.sampling import (
    apply_temperature,
    generate,
    sample_next,
    top_k_filter,
    top_p_filter,
)
from underhood.model.attention import KVCache
from underhood.model.gpt import GPT

PEAKED = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
# a distribution whose cumulative mass is 0.50, 0.75, 0.90, 1.00 — no row below lands on a boundary
NUCLEUS = torch.tensor([[0.50, 0.25, 0.15, 0.10]]).log()


def test_apply_temperature_sharpens_below_one_and_flattens_above() -> None:
    peak = {t: apply_temperature(PEAKED, t).softmax(-1).max().item() for t in (0.5, 1.0, 2.0)}
    assert peak[0.5] > peak[1.0] > peak[2.0]
    assert peak[1.0] == pytest.approx(PEAKED.softmax(-1).max().item())


@pytest.mark.parametrize(
    "temperature",
    [pytest.param(0.0, id="zero"), pytest.param(-1.0, id="negative")],
)
def test_apply_temperature_rejects_non_positive(temperature: float) -> None:
    with pytest.raises(ValueError):
        apply_temperature(PEAKED, temperature)


@pytest.mark.parametrize(
    ("k", "kept"),
    [
        pytest.param(1, 1, id="k of one is argmax"),
        pytest.param(3, 3, id="k inside the vocab"),
        pytest.param(4, 4, id="k is the whole vocab"),
        pytest.param(9, 4, id="k larger than the vocab keeps everything"),
    ],
)
def test_top_k_keeps_exactly_k(k: int, kept: int) -> None:
    filtered = top_k_filter(PEAKED, k)
    assert int(filtered.isfinite().sum().item()) == kept


def test_top_k_keeps_the_largest_logits_unchanged() -> None:
    filtered = top_k_filter(PEAKED, 2)
    assert torch.equal(filtered[0, :2], PEAKED[0, :2])
    assert bool(filtered[0, 2:].isinf().all())


@pytest.mark.parametrize("k", [pytest.param(0, id="zero"), pytest.param(-2, id="negative")])
def test_top_k_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError):
        top_k_filter(PEAKED, k)


@pytest.mark.parametrize(
    ("p", "kept"),
    [
        pytest.param(0.40, 1, id="p under the top token's own mass"),
        pytest.param(0.60, 2, id="p needing a second token"),
        pytest.param(0.80, 3, id="p needing a third"),
        pytest.param(0.95, 4, id="p needing the tail as well"),
    ],
)
def test_top_p_keeps_the_smallest_set_that_reaches_p(p: float, kept: int) -> None:
    filtered = top_p_filter(NUCLEUS, p)
    assert int(filtered.isfinite().sum().item()) == kept


def test_top_p_always_keeps_at_least_one_token() -> None:
    assert int(top_p_filter(NUCLEUS, 0.01).isfinite().sum().item()) == 1


@pytest.mark.parametrize(
    "p",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-0.1, id="negative"),
        pytest.param(1.5, id="above one"),
    ],
)
def test_top_p_rejects_p_outside_the_unit_interval(p: float) -> None:
    with pytest.raises(ValueError):
        top_p_filter(NUCLEUS, p)


def test_sample_next_with_top_k_one_is_argmax() -> None:
    logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 0.0, 0.5]])
    assert torch.equal(sample_next(logits, top_k=1, top_p=None), torch.tensor([[1], [0]]))


def test_sample_next_is_reproducible_given_a_generator() -> None:
    logits = torch.randn(4, 12)
    draws = [
        sample_next(logits, generator=torch.Generator().manual_seed(7)),
        sample_next(logits, generator=torch.Generator().manual_seed(7)),
    ]
    assert torch.equal(draws[0], draws[1])
    assert draws[0].shape == (4, 1)


def test_generate_extends_the_prompt_without_touching_it(tiny_gpt: GPT, dims: TinyDims) -> None:
    prompt = torch.randint(0, dims.vocab_size, (2, 3))
    out = generate(tiny_gpt, prompt, 5, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], prompt)


def test_generate_crops_the_context_to_the_block(make_gpt: GptFactory, dims: TinyDims) -> None:
    model = make_gpt(block_size=4)
    prompt = torch.randint(0, dims.vocab_size, (1, 2))
    out = generate(model, prompt, 10, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 12), "generating past block_size must crop, not raise"


def test_generate_stays_inside_the_vocabulary(tiny_gpt: GPT, dims: TinyDims) -> None:
    prompt = torch.zeros((1, 1), dtype=torch.long)
    out = generate(tiny_gpt, prompt, 20, generator=torch.Generator().manual_seed(0))
    assert int(out.min().item()) >= 0
    assert int(out.max().item()) < dims.vocab_size


def test_low_temperature_concentrates_the_output(tiny_gpt: GPT, dims: TinyDims) -> None:
    prompt = torch.zeros((1, 1), dtype=torch.long)
    seeded = torch.Generator().manual_seed(0)
    cold = generate(tiny_gpt, prompt, 40, temperature=0.05, generator=seeded)
    hot = generate(tiny_gpt, prompt, 40, temperature=2.0, generator=seeded)
    distinct = (len(set(cold[0].tolist())), len(set(hot[0].tolist())))
    assert distinct[0] < distinct[1], f"cold then hot produced {distinct} distinct tokens"


def test_a_cache_changes_nothing_but_the_speed(tiny_gpt: GPT, dims: TinyDims) -> None:
    """Greedy on both sides, so the cache is the only thing that could explain a difference."""
    prompt = torch.randint(0, dims.vocab_size, (2, 2))
    plain = generate(tiny_gpt, prompt, 6, top_k=1, top_p=None)
    cached = generate(
        tiny_gpt, prompt, 6, top_k=1, top_p=None, cache=KVCache(n_layers=dims.n_layers)
    )
    assert torch.equal(plain, cached), f"cached {cached.tolist()} != plain {plain.tolist()}"


def test_a_cached_prompt_goes_through_in_one_prefill_pass(
    tiny_gpt: GPT, dims: TinyDims, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason GPT.forward takes a cache: the prompt need not be fed a token at a time."""
    widths: list[int] = []
    original = tiny_gpt.forward

    def counting(
        idx: torch.Tensor, targets: torch.Tensor | None = None, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        widths.append(idx.size(1))
        return original(idx, targets, cache)

    monkeypatch.setattr(tiny_gpt, "forward", counting)
    prompt = torch.randint(0, dims.vocab_size, (1, 4))
    generate(tiny_gpt, prompt, 3, top_k=1, top_p=None, cache=KVCache(n_layers=dims.n_layers))
    assert widths == [4, 1, 1], f"expected one prefill then single steps, got {widths}"


def test_without_a_cache_every_step_re_reads_the_context(
    tiny_gpt: GPT, dims: TinyDims, monkeypatch: pytest.MonkeyPatch
) -> None:
    widths: list[int] = []
    original = tiny_gpt.forward

    def counting(
        idx: torch.Tensor, targets: torch.Tensor | None = None, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        widths.append(idx.size(1))
        return original(idx, targets, cache)

    monkeypatch.setattr(tiny_gpt, "forward", counting)
    generate(tiny_gpt, torch.randint(0, dims.vocab_size, (1, 4)), 3, top_k=1, top_p=None)
    assert widths == [4, 5, 6], "the uncached path grows its context by one every step"


def test_a_checkpoint_loads_back_as_the_model_and_tokenizer_that_wrote_it(
    make_gpt: GptFactory, dims: TinyDims, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gpt.pt"
    monkeypatch.setattr(sampling, "CHECKPOINT", path)
    model = make_gpt()
    merges, vocab = {(104, 105): 256}, {i: bytes([i]) for i in range(256)} | {256: b"hi"}
    torch.save(
        {
            "block_size": dims.block_size,
            "merges": merges,
            "model": model.state_dict(),
            "vocab": vocab,
            "vocab_size": dims.vocab_size,
        },
        path,
    )
    monkeypatch.setattr(sampling, "GPT", lambda **kw: type(model)(**{**dims._asdict(), **kw}))

    tokenizer, loaded = sampling._load(torch.device("cpu"))

    assert (tokenizer.merges, tokenizer.vocab) == (merges, vocab)
    assert not loaded.training, "loaded for inference"
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name


def test_loading_without_a_checkpoint_says_how_to_make_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sampling, "CHECKPOINT", tmp_path / "missing.pt")

    with pytest.raises(FileNotFoundError, match="underhood-train"):
        sampling._load(torch.device("cpu"))
