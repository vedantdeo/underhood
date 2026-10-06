"""Attention from scratch.

The tests in tests/model/test_attention.py compare your implementation against
torch.nn.functional.scaled_dot_product_attention, so you are checking your understanding against
the real thing, not against my reading of it. Order: causal_mask, single_head_attention, then
MultiHeadAttention.forward.

KVCache lives here too: it holds nothing but keys and values, and attention is the only
operation in a transformer that has any use for another position's.

Reference: "The Illustrated Transformer", then the attention section of Karpathy's "Let's build
GPT".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def causal_mask(t_q: int, t_k: int | None = None, device: torch.device | None = None) -> Tensor:
    """A (t_q, t_k) boolean mask: True where a query may attend to a key.

    Square and lower-triangular when t_k is omitted. A longer t_k means the leading keys are
    cached positions, which every query may see, so the diagonal shifts right by the difference —
    and a single decoding step, t_q of 1, comes out all True, as it should.
    """
    if t_k is None:
        t_k = t_q
    return torch.ones(t_q, t_k, device=device, dtype=torch.bool).tril(diagonal=t_k - t_q)


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
        mask = causal_mask(scores.size(-2), scores.size(-1), device=scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v


class KVCache:
    """The keys and values of every position seen so far, one slot per layer.

    Only k and v are kept: a query has no use after it has attended, so there is nothing to store.
    """

    keys: list[Tensor | None]
    values: list[Tensor | None]

    def __init__(self, n_layers: int) -> None:
        self.keys = [None] * n_layers
        self.values = [None] * n_layers

    @property
    def t(self) -> int:
        """How many positions are cached — 0 before anything has been appended."""
        first = self.keys[0]
        return 0 if first is None else first.size(-2)

    def append(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Add this call's k and v for one layer and return that layer's whole history.

        k and v: (batch, n_heads, t_new, d_head) — t_new is the prompt length while prefilling and
        1 per step after. Returned: (batch, n_heads, t_total, d_head), the new positions last.
        """
        held_k, held_v = self.keys[layer], self.values[layer]
        keys = k if held_k is None else torch.cat((held_k, k), dim=-2)
        values = v if held_v is None else torch.cat((held_v, v), dim=-2)
        self.keys[layer], self.values[layer] = keys, values
        return keys, values


class MultiHeadAttention(nn.Module):
    """Split d_model into n_heads heads, attend in each, concatenate, project.

    Keep these attribute names; the tests use them to build a reference from your own weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        causal: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cache: KVCache | None = None, layer: int = 0) -> Tensor:
        """x: (batch, t, d_model) -> (batch, t, d_model).

        Project to q, k, v; reshape each to (batch, n_heads, t, d_head); attend per head with the
        causal flag; merge heads back to (batch, t, d_model); apply out_proj and drop. Given a
        cache, this layer's k and v are appended to it first and the queries attend over the whole
        history; the mask widens to match, so nothing else changes.
        """
        b, t, _ = x.shape
        q, k, v = (
            proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        if cache is not None:
            k, v = cache.append(layer, k, v)
        heads = single_head_attention(q, k, v, causal=self.causal, dropout=self.attn_dropout)
        merged = heads.transpose(1, 2).reshape(b, t, self.d_model)
        return self.resid_dropout(self.out_proj(merged))
