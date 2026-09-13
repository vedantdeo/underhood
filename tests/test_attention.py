"""Specification for week01/attention.py, checked against torch's own attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from underhood.week01.attention import MultiHeadAttention, causal_mask, single_head_attention

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
