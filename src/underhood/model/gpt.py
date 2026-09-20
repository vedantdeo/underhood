"""GPT from scratch: the per-position MLP, the block that wraps it, and the model that stacks them.

The tests in tests/model/test_gpt.py are the specification. Make them green one class at a time:
FeedForward, then Block, then GPT. Attention is already yours — import MultiHeadAttention rather
than writing it again.

Reference: Karpathy's "Let's build GPT". Do not read reference/ng_video_lecture_gpt.py or
reference/nanogpt_model.py until yours passes; then read both and compare.
"""

from __future__ import annotations

from torch import Tensor, nn

from underhood import config
from underhood.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """The per-position MLP: up to 4 * d_model, a non-linearity, back down, then dropout.

    Every position goes through it independently — this is where the model thinks about a token,
    as opposed to attention, which is where tokens look at each other.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, t, d_model) -> (batch, t, d_model)."""
        raise NotImplementedError


class Block(nn.Module):
    """One transformer block: pre-norm attention and pre-norm feed-forward, each residual.

    Pre-norm means the LayerNorm sits inside the residual branch, not around it, so the residual
    path stays a clean sum from input to output. Keep these four attribute names; the tests zero
    the two sublayers to check the residual survives.
    """

    ln1: nn.LayerNorm
    attn: MultiHeadAttention
    ln2: nn.LayerNorm
    ff: FeedForward

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """x + attn(ln1(x)), then that + ff(ln2(x)). Shape in, same shape out."""
        raise NotImplementedError


class GPT(nn.Module):
    """Token and position embeddings, n_layers blocks, a final norm, a projection back to vocab.

    Keep these attribute names: the tests read them, and the KV cache in inference/kv_cache.py
    walks `blocks` one at a time rather than calling `forward`.
    """

    block_size: int
    tok_emb: nn.Embedding
    pos_emb: nn.Embedding
    blocks: nn.ModuleList
    ln_f: nn.LayerNorm
    lm_head: nn.Linear

    def __init__(
        self,
        vocab_size: int = config.VOCAB_SIZE,
        block_size: int = config.BLOCK_SIZE,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        n_layers: int = config.N_LAYERS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """idx: (batch, t) token ids -> logits (batch, t, vocab_size), and a loss if targets given.

        Positions are 0..t-1, so t must not exceed block_size — raise ValueError when it does,
        rather than letting the embedding lookup fail somewhere deeper. With targets (batch, t),
        the loss is cross-entropy over batch and time flattened together.
        """
        raise NotImplementedError
