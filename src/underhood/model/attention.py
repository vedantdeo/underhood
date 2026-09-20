"""Attention from scratch.

The tests in tests/model/test_attention.py compare your implementation against
torch.nn.functional.scaled_dot_product_attention, so you are checking your understanding against
the real thing, not against my reading of it. Order: causal_mask, single_head_attention, then
MultiHeadAttention.forward.

Reference: "The Illustrated Transformer", then the attention section of Karpathy's "Let's build
GPT".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def causal_mask(t: int, device: torch.device | None = None) -> Tensor:
    """A (t, t) boolean mask: True where position i may attend to position j, i.e. j <= i."""
    return torch.tril(torch.ones(t, t, device=device, dtype=torch.bool))


def single_head_attention(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = False, dropout: nn.Module | None = None
) -> Tensor:
    """Scaled dot-product attention for one head.

    q, k, v: (batch, t, d). Returns (batch, t, d). scores = q @ k^T / sqrt(d); mask future
    positions with -inf if causal; softmax over the last axis; weighted sum of v. `dropout` takes
    a module rather than a rate, so it follows its parent's train/eval state.
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d**0.5)
    if causal:
        mask = causal_mask(scores.size(-1), device=scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v


class MultiHeadAttention(nn.Module):
    """Split d_model into n_heads heads, attend in each, concatenate, project.

    Keep these attribute names; the tests use them to build a reference from your own weights.
    """

    def __init__(
        self, d_model: int, n_heads: int, causal: bool = True, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, t, d_model) -> (batch, t, d_model).

        Project to q, k, v; reshape each to (batch, n_heads, t, d_head); attend per head with the
        causal flag; merge heads back to (batch, t, d_model); apply out_proj and drop.
        """
        b, t, _ = x.shape
        q, k, v = (
            proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        heads = single_head_attention(q, k, v, causal=self.causal, dropout=self.attn_dropout)
        merged = heads.transpose(1, 2).reshape(b, t, self.d_model)
        return self.resid_dropout(self.out_proj(merged))
