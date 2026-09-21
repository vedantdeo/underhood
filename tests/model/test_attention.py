"""Specification for model/attention.py, checked against torch's own attention.

KVCache is specified here too, since that is where it lives and what it is for.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tests.conftest import TinyDims
from underhood.model.attention import (
    KVCache,
    MultiHeadAttention,
    causal_mask,
    single_head_attention,
)

torch.manual_seed(0)


def test_causal_mask_is_lower_triangular() -> None:
    expected = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    assert torch.equal(causal_mask(3), expected)
    assert causal_mask(5).dtype == torch.bool


def test_single_head_matches_torch_without_mask() -> None:
    q, k, v = (torch.randn(2, 7, 16) for _ in range(3))
    assert torch.allclose(
        single_head_attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-5
    )


def test_single_head_matches_torch_with_causal_mask() -> None:
    q, k, v = (torch.randn(2, 7, 16) for _ in range(3))
    mine = single_head_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(mine, ref, atol=1e-5)


def test_single_head_rows_are_weighted_averages_of_v() -> None:
    q, k = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    v = torch.ones(1, 4, 8)  # if every value is 1, any convex combination is 1
    assert torch.allclose(single_head_attention(q, k, v), torch.ones(1, 4, 8), atol=1e-6)


def test_multi_head_output_shape() -> None:
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    assert mha(torch.randn(3, 5, 32)).shape == (3, 5, 32)


def test_multi_head_is_causal() -> None:
    mha = MultiHeadAttention(d_model=32, n_heads=4, causal=True)
    x = torch.randn(1, 6, 32)
    y = mha(x)
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn(1, 2, 32)  # change the last two positions only
    y2 = mha(x2)
    assert torch.allclose(y[:, :4], y2[:, :4], atol=1e-6), "earlier positions saw the future"
    assert not torch.allclose(y[:, 4:], y2[:, 4:])


def test_multi_head_matches_reference_built_from_its_own_weights() -> None:
    d_model, n_heads, t = 32, 4, 6
    mha = MultiHeadAttention(d_model, n_heads, causal=True)
    x = torch.randn(2, t, d_model)

    q = mha.q_proj(x).view(2, t, n_heads, d_model // n_heads).transpose(1, 2)
    k = mha.k_proj(x).view(2, t, n_heads, d_model // n_heads).transpose(1, 2)
    v = mha.v_proj(x).view(2, t, n_heads, d_model // n_heads).transpose(1, 2)
    heads = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = mha.out_proj(heads.transpose(1, 2).reshape(2, t, d_model))

    assert torch.allclose(mha(x), ref, atol=1e-5)


def test_multi_head_rejects_indivisible_dims() -> None:
    try:
        MultiHeadAttention(d_model=30, n_heads=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for d_model not divisible by n_heads")


def test_multi_head_has_the_two_dropout_sites_gpt2_used() -> None:
    """GPT-2 drops the attention weights and the output projection, so two per attention."""
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.1)
    sites = [module for module in mha.modules() if isinstance(module, nn.Dropout)]
    assert len(sites) == 2, f"expected weights and residual dropout, found {len(sites)}"


def test_multi_head_dropout_is_off_in_eval() -> None:
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.5).eval()
    x = torch.randn(2, 6, 32)
    assert torch.equal(mha(x), mha(x)), "eval() must be deterministic however high dropout is"


def test_multi_head_dropout_perturbs_training() -> None:
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.5).train()
    x = torch.randn(2, 6, 32)
    assert not torch.equal(mha(x), mha(x)), "train() with dropout must vary between calls"


def test_multi_head_is_deterministic_in_training_without_dropout() -> None:
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0).train()
    x = torch.randn(2, 6, 32)
    assert torch.equal(mha(x), mha(x)), "a rate of zero must be a genuine no-op"


@pytest.mark.parametrize(
    ("t_q", "t_k", "expected"),
    [
        pytest.param(3, 3, [[1, 0, 0], [1, 1, 0], [1, 1, 1]], id="square is the plain causal mask"),
        pytest.param(
            2, 5, [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], id="two new tokens onto three cached"
        ),
        pytest.param(1, 6, [[1, 1, 1, 1, 1, 1]], id="one decoding step sees all six cached"),
    ],
)
def test_causal_mask_shifts_by_the_cached_length(
    t_q: int, t_k: int, expected: list[list[int]]
) -> None:
    assert torch.equal(causal_mask(t_q, t_k), torch.tensor(expected, dtype=torch.bool))


def test_causal_mask_is_square_by_default() -> None:
    assert torch.equal(causal_mask(4), causal_mask(4, 4))


def test_attending_one_position_at_a_time_matches_one_full_pass() -> None:
    mha = MultiHeadAttention(d_model=32, n_heads=4, causal=True).eval()
    x = torch.randn(2, 5, 32)
    cache = KVCache(n_layers=1)
    stepped = torch.cat([mha(x[:, i : i + 1], cache, 0) for i in range(x.size(1))], dim=1)
    assert torch.allclose(mha(x), stepped, atol=1e-5)


def test_prefilling_then_stepping_matches_one_full_pass() -> None:
    mha = MultiHeadAttention(d_model=32, n_heads=4, causal=True).eval()
    x = torch.randn(2, 5, 32)
    cache = KVCache(n_layers=1)
    prefilled = mha(x[:, :3], cache, 0)
    stepped = torch.cat([mha(x[:, i : i + 1], cache, 0) for i in (3, 4)], dim=1)
    assert torch.allclose(mha(x), torch.cat((prefilled, stepped), dim=1), atol=1e-5)


def _heads(dims: TinyDims, t: int) -> Tensor:
    return torch.randn(2, dims.n_heads, t, dims.d_model // dims.n_heads)


def test_cache_starts_empty(dims: TinyDims) -> None:
    cache = KVCache(n_layers=dims.n_layers)
    assert cache.t == 0
    assert cache.keys == [None] * dims.n_layers
    assert cache.values == [None] * dims.n_layers


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(1, id="one position"),
        pytest.param(3, id="three positions"),
        pytest.param(8, id="a full block"),
    ],
)
def test_append_grows_the_cache_one_position_at_a_time(dims: TinyDims, steps: int) -> None:
    cache = KVCache(n_layers=1)
    appended = [_heads(dims, 1) for _ in range(steps)]
    for k in appended[:-1]:
        cache.append(0, k, _heads(dims, 1))
    keys, _ = cache.append(0, appended[-1], _heads(dims, 1))
    assert cache.t == steps
    assert keys.shape == (2, dims.n_heads, steps, dims.d_model // dims.n_heads)
    assert torch.allclose(keys[:, :, -1:], appended[-1]), "the newest position goes last"


def test_a_prompt_appended_at_once_matches_one_position_at_a_time(dims: TinyDims) -> None:
    """Prefill: the cache cannot tell whether positions arrived together or one by one."""
    ks = [_heads(dims, 1) for _ in range(5)]
    vs = [_heads(dims, 1) for _ in range(5)]

    stepped = KVCache(n_layers=1)
    for k, v in zip(ks, vs, strict=True):
        stepped.append(0, k, v)
    prefilled, _ = KVCache(n_layers=1).append(0, torch.cat(ks, dim=-2), torch.cat(vs, dim=-2))

    kept = stepped.keys[0]
    assert kept is not None
    assert torch.equal(kept, prefilled)
    assert stepped.t == 5


def test_layers_are_kept_apart(dims: TinyDims) -> None:
    cache = KVCache(n_layers=dims.n_layers)
    cache.append(0, _heads(dims, 1), _heads(dims, 1))
    assert cache.keys[1] is None, "appending to one layer must not touch another"
