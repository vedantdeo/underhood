"""Specification for model/pretrained.py: GPT-2's two switches, then its weights in your GPT.

Every test builds a randomly initialised Hugging Face GPT-2 at the tiny dimensions, so nothing is
downloaded; `uv run underhood-gpt2-check` is where the real checkpoint gets compared.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Config, GPT2LMHeadModel

from tests.conftest import TinyDims
from underhood.model.attention import MultiHeadAttention
from underhood.model.gpt import GPT, FeedForward, GeluApproximate
from underhood.model.pretrained import gpt2_state_dict, gpt_for


@pytest.fixture
def hf_gpt2(dims: TinyDims) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    hf_config = GPT2Config(
        vocab_size=dims.vocab_size,
        n_positions=dims.block_size,
        n_embd=dims.d_model,
        n_layer=dims.n_layers,
        n_head=dims.n_heads,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(hf_config).eval()
    with torch.no_grad():  # GPT-2's own init is too small to tell q from k, or a gain from a bias
        for param in model.parameters():
            param.normal_(0.0, 0.3)
    return model


@pytest.fixture
def loaded(hf_gpt2: GPT2LMHeadModel) -> GPT:
    model = gpt_for(hf_gpt2.config)
    model.load_state_dict(gpt2_state_dict(hf_gpt2.state_dict()), strict=True)
    return model.eval()


@pytest.fixture
def tokens(dims: TinyDims) -> Tensor:
    torch.manual_seed(1)
    return torch.randint(0, dims.vocab_size, (3, dims.block_size))


@pytest.mark.parametrize(
    ("bias", "has_bias"),
    [
        pytest.param({}, False, id="no attention bias by default, as before"),
        pytest.param({"bias": True}, True, id="GPT-2's biases on q, k, v and out"),
    ],
)
def test_attention_projections_take_a_bias_switch(bias: dict[str, bool], has_bias: bool) -> None:
    attn = MultiHeadAttention(16, 4, **bias)
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
        assert (proj.bias is not None) is has_bias, proj


@pytest.mark.parametrize(
    ("approximate", "kwargs"),
    [
        pytest.param("none", {}, id="exact GELU by default, as before"),
        pytest.param("tanh", {"gelu_approximate": "tanh"}, id="GPT-2's tanh approximation"),
    ],
)
def test_feed_forward_takes_the_gelu_approximation(
    approximate: GeluApproximate, kwargs: dict[str, GeluApproximate]
) -> None:
    torch.manual_seed(0)
    ff = FeedForward(16, 0.0, **kwargs)
    x = 3 * torch.randn(2, 5, 16)
    up, down = cast(torch.nn.Linear, ff.net[0]), cast(torch.nn.Linear, ff.net[2])
    expected = down(F.gelu(up(x), approximate=approximate))
    assert torch.allclose(ff(x), expected, atol=1e-6), approximate


def test_gpt_passes_both_switches_down_to_every_block(dims: TinyDims) -> None:
    model = GPT(**dims._asdict(), bias=True, gelu_approximate="tanh")
    bias_counts = sum(1 for name, _ in model.named_parameters() if "proj.bias" in name)
    assert bias_counts == 4 * dims.n_layers


def test_the_converted_weights_load_strictly(hf_gpt2: GPT2LMHeadModel) -> None:
    """Every key named, every shape right, nothing left over."""
    converted = gpt2_state_dict(hf_gpt2.state_dict())
    model = gpt_for(hf_gpt2.config)
    expected = model.state_dict()
    assert sorted(converted) == sorted(expected)
    for name, tensor in converted.items():
        assert tensor.shape == expected[name].shape, name


def test_your_logits_are_hugging_faces(
    hf_gpt2: GPT2LMHeadModel, loaded: GPT, tokens: Tensor
) -> None:
    with torch.no_grad():
        theirs = cast(Tensor, hf_gpt2(tokens).logits)
        mine, _ = loaded(tokens)
    gap = (theirs - mine).abs().max().item()
    assert torch.allclose(mine, theirs, atol=1e-5, rtol=1e-4), f"largest gap {gap:.2e}"


def test_your_loss_is_hugging_faces(hf_gpt2: GPT2LMHeadModel, loaded: GPT, tokens: Tensor) -> None:
    """Hugging Face shifts labels inside the model; yours takes targets already shifted."""
    with torch.no_grad():
        theirs = cast(Tensor, hf_gpt2(tokens, labels=tokens).loss)
        _, mine = loaded(tokens[:, :-1], tokens[:, 1:])
    assert mine is not None
    assert torch.allclose(mine, theirs, atol=1e-5), (mine.item(), theirs.item())
