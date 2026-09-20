"""Specification for model/gpt.py: the MLP, the block, then the model that stacks them."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import GptFactory, TinyDims
from underhood.model.gpt import GPT, Block, FeedForward

D_MODEL, N_HEADS = 16, 4


def test_feedforward_preserves_shape() -> None:
    ff = FeedForward(d_model=D_MODEL, dropout=0.0)
    assert ff(torch.randn(3, 5, D_MODEL)).shape == (3, 5, D_MODEL)


def test_feedforward_expands_fourfold_inside() -> None:
    ff = FeedForward(d_model=D_MODEL, dropout=0.0)
    hidden = 4 * D_MODEL
    up = D_MODEL * hidden + hidden
    down = hidden * D_MODEL + D_MODEL
    assert sum(p.numel() for p in ff.parameters()) == up + down


def test_feedforward_treats_positions_independently() -> None:
    ff = FeedForward(d_model=D_MODEL, dropout=0.0).eval()
    x = torch.randn(1, 4, D_MODEL)
    changed = x.clone()
    changed[:, 3] = torch.randn(D_MODEL)
    assert torch.allclose(ff(x)[:, :3], ff(changed)[:, :3], atol=1e-6)


def test_block_preserves_shape() -> None:
    block = Block(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0).eval()
    assert block(torch.randn(2, 6, D_MODEL)).shape == (2, 6, D_MODEL)


def test_block_is_a_residual_around_its_sublayers() -> None:
    block = Block(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0).eval()
    with torch.no_grad():
        for sublayer in (block.attn, block.ff):
            for parameter in sublayer.parameters():
                parameter.zero_()
    x = torch.randn(2, 5, D_MODEL)
    assert torch.allclose(block(x), x, atol=1e-6), "a zeroed block must pass its input straight on"


def test_block_is_causal() -> None:
    block = Block(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0).eval()
    x = torch.randn(1, 6, D_MODEL)
    changed = x.clone()
    changed[:, 4:] = torch.randn(1, 2, D_MODEL)
    assert torch.allclose(block(x)[:, :4], block(changed)[:, :4], atol=1e-6)


@pytest.mark.parametrize(
    ("batch", "t"),
    [
        pytest.param(1, 1, id="a single token"),
        pytest.param(4, 3, id="a short batch"),
        pytest.param(2, 8, id="a context filling the block exactly"),
    ],
)
def test_gpt_logits_have_a_row_per_position(
    tiny_gpt: GPT, dims: TinyDims, batch: int, t: int
) -> None:
    idx = torch.randint(0, dims.vocab_size, (batch, t))
    logits, loss = tiny_gpt(idx)
    assert logits.shape == (batch, t, dims.vocab_size)
    assert loss is None, "no targets, no loss"


def test_gpt_loss_is_cross_entropy_over_batch_and_time(tiny_gpt: GPT, dims: TinyDims) -> None:
    idx = torch.randint(0, dims.vocab_size, (3, 5))
    targets = torch.randint(0, dims.vocab_size, (3, 5))
    logits, loss = tiny_gpt(idx, targets)
    assert loss is not None
    expected = F.cross_entropy(logits.reshape(-1, dims.vocab_size), targets.reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gpt_loss_at_init_is_roughly_uniform(tiny_gpt: GPT, dims: TinyDims) -> None:
    idx = torch.randint(0, dims.vocab_size, (8, 8))
    targets = torch.randint(0, dims.vocab_size, (8, 8))
    _, loss = tiny_gpt(idx, targets)
    assert loss is not None
    uniform = math.log(dims.vocab_size)
    detail = f"{loss.item():.3f} is not near ln(vocab) = {uniform:.3f}"
    assert abs(loss.item() - uniform) < 0.7, detail


def test_gpt_is_causal(tiny_gpt: GPT, dims: TinyDims) -> None:
    idx = torch.randint(0, dims.vocab_size, (1, 6))
    changed = idx.clone()
    changed[:, 4:] = (idx[:, 4:] + 1) % dims.vocab_size
    before, _ = tiny_gpt(idx)
    after, _ = tiny_gpt(changed)
    assert torch.allclose(before[:, :4], after[:, :4], atol=1e-6), "a position saw the future"
    assert not torch.allclose(before[:, 4:], after[:, 4:])


def test_gpt_rejects_a_context_longer_than_its_block(tiny_gpt: GPT, dims: TinyDims) -> None:
    too_long = torch.randint(0, dims.vocab_size, (1, dims.block_size + 1))
    with pytest.raises(ValueError):
        tiny_gpt(too_long)


def test_gpt_stacks_the_layers_it_was_asked_for(make_gpt: GptFactory) -> None:
    assert len(make_gpt(n_layers=3).blocks) == 3
