"""Spec for inference/quantization.py: the arithmetic under the benchmark, on toy inputs.

Nothing here loads a model; `uv run underhood-quant-bench` is where the real ones run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("mlx")  # Mac-only; CI runs on Linux

import mlx.core as mx  # noqa: E402
from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: E402

from underhood.inference.quantization import (  # noqa: E402
    PROMPTS,
    chat_tokens,
    first_divergence,
    kl_divergence,
    load_prompts,
    log_probs,
    time_stream,
    top1_agreement,
    variant_dir,
)


def test_the_prompts_are_ten_in_canonical_order() -> None:
    lines = PROMPTS.read_text(encoding="utf-8").splitlines()
    assert [list(json.loads(line)) for line in lines] == [["id", "text"]] * 10
    ids = [prompt.id for prompt in load_prompts()]
    assert ids == sorted(set(ids)), ids


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _stream(clock: FakeClock, delays: Sequence[float]) -> Iterator[int]:
    """Tokens 0, 1, 2..., each arriving `delay` seconds after the one before."""
    for token, delay in enumerate(delays):
        clock.now += delay
        yield token


@pytest.mark.parametrize(
    ("delays", "ttft", "decode_tps"),
    [
        pytest.param([0.5, 0.1, 0.1, 0.1], 0.5, 10.0, id="a prefill, then ten tokens a second"),
        pytest.param([0.25], 0.25, None, id="a single token has no decode rate"),
    ],
)
def test_time_stream(delays: list[float], ttft: float, decode_tps: float | None) -> None:
    clock = FakeClock()
    timed = time_stream(_stream(clock, delays), clock=clock)
    assert timed.tokens == tuple(range(len(delays)))
    assert timed.ttft_s == pytest.approx(ttft), timed
    assert timed.decode_tps == (None if decode_tps is None else pytest.approx(decode_tps)), timed


def test_time_stream_refuses_an_empty_stream() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        time_stream(iter([]))


def test_log_probs_row_i_is_the_distribution_that_predicted_token_i() -> None:
    vocab = 5

    def echo(inputs: mx.array) -> mx.array:
        """A model whose logits at each position spike on the token sitting there."""
        return mx.eye(vocab)[inputs] * 10

    rows = log_probs(echo, prompt=[1, 2], continuation=[3, 4, 0])
    assert rows.shape == (3, vocab)
    assert rows.argmax(axis=-1).tolist() == [2, 3, 4], "the last prompt token, then all but last"
    assert mx.allclose(mx.exp(rows).sum(axis=-1), mx.ones(3)).item()


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        pytest.param([0.5, 0.5], [0.5, 0.5], 0.0, id="identical distributions"),
        pytest.param(
            [0.5, 0.5],
            [0.9, 0.1],
            0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1),
            id="a skewed other",
        ),
        pytest.param(
            [0.9, 0.1],
            [0.5, 0.5],
            0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5),
            id="the same pair reversed, which is a different number",
        ),
    ],
)
def test_kl_divergence(p: list[float], q: list[float], expected: float) -> None:
    got = kl_divergence(mx.log(mx.array([p])), mx.log(mx.array([q])))
    assert got.shape == (1,)
    assert got.item() == pytest.approx(expected, abs=1e-6)


def test_top1_agreement_counts_the_positions_whose_argmax_matches() -> None:
    reference = mx.log(mx.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]))
    other = mx.log(mx.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.1, 0.1, 0.8]]))
    assert top1_agreement(reference, other) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param([1, 2, 3], [1, 2, 3], None, id="identical generations"),
        pytest.param([1, 2, 3], [9, 2, 3], 0, id="different from the first token"),
        pytest.param([1, 2, 3], [1, 2, 9], 2, id="different only at the last token"),
        pytest.param([1, 2], [1, 2, 3], 2, id="one stopped before the other"),
        pytest.param([], [], None, id="both empty"),
    ],
)
def test_first_divergence(a: list[int], b: list[int], expected: int | None) -> None:
    assert first_divergence(a, b) == expected


@pytest.mark.parametrize(
    ("repo", "bits", "name"),
    [
        pytest.param(
            "mlx-community/Llama-3.2-3B-Instruct-bf16",
            4,
            "Llama-3.2-3B-Instruct-q4",
            id="the -bf16 suffix is dropped",
        ),
        pytest.param(
            "mlx-community/SmolLM2-135M-Instruct",
            8,
            "SmolLM2-135M-Instruct-q8",
            id="a repo with no suffix keeps its name",
        ),
    ],
)
def test_variant_dir(repo: str, bits: int, name: str, tmp_path: Path) -> None:
    assert variant_dir(repo, bits, tmp_path) == tmp_path / name


class _Tokenizer:
    """A chat template and an encoder, recording whether it was asked to add special tokens."""

    def __init__(self, bos_token: str | None, template: str) -> None:
        self.bos_token, self.template = bos_token, template
        self.added: list[bool] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize and add_generation_prompt
        return self.template.format(messages[0]["content"])

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        self.added.append(add_special_tokens)
        return [ord(c) for c in text]


@pytest.mark.parametrize(
    ("bos", "template", "adds"),
    [
        pytest.param(
            "<s>", "<s>[INST] {} [/INST]", False, id="the template already starts with BOS"
        ),
        pytest.param("<s>", "[INST] {} [/INST]", True, id="the template leaves BOS out"),
        pytest.param(None, "<|user|>{}", True, id="a tokenizer with no BOS at all"),
    ],
)
def test_chat_tokens_adds_bos_exactly_once(bos: str | None, template: str, adds: bool) -> None:
    tokenizer = _Tokenizer(bos, template)

    ids = chat_tokens(cast(TokenizerWrapper, tokenizer), "hi")

    assert tokenizer.added == [adds]
    assert ids == [ord(c) for c in template.format("hi")]
