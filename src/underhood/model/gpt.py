"""GPT from scratch: the per-position MLP, the block that wraps it, and the model that stacks them.

The tests in tests/model/test_gpt.py are the specification. Make them green one class at a time:
FeedForward, then Block, then GPT. Attention is already yours — import MultiHeadAttention rather
than writing it again.

Reference: Karpathy's "Let's build GPT". Do not read reference/ng_video_lecture_gpt.py or
reference/nanogpt_model.py until yours passes; then read both and compare.
"""

from __future__ import annotations

from typing import Literal, cast

from torch import Tensor, arange, nn

from underhood import config
from underhood.model.attention import KVCache, MultiHeadAttention

GeluApproximate = Literal["none", "tanh"]  # the only values F.gelu accepts


class FeedForward(nn.Module):
    """The per-position MLP: up to 4 * d_model, a non-linearity, back down, then dropout.

    Every position goes through it independently — this is where the model thinks about a token,
    as opposed to attention, which is where tokens look at each other.
    """

    def __init__(
        self, d_model: int, dropout: float, gelu_approximate: GeluApproximate = "none"
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate=gelu_approximate),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, t, d_model) -> (batch, t, d_model)."""
        return self.net(x)


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

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        bias: bool = False,
        gelu_approximate: GeluApproximate = "none",
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True, dropout=dropout, bias=bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout, gelu_approximate=gelu_approximate)

    def forward(self, x: Tensor, cache: KVCache | None = None, layer: int = 0) -> Tensor:
        """x + attn(ln1(x)), then that + ff(ln2(x)). Shape in, same shape out.

        The cache and layer index are handed straight to attention, the only sublayer with any
        memory of other positions.
        """
        x = x + self.attn(self.ln1(x), cache, layer)
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """Token and position embeddings, n_layers blocks, a final norm, a projection back to vocab.

    Keep these attribute names; the tests read them. Pass a KVCache to forward and the model
    extends a sequence it has already seen instead of reading one from scratch — which is the
    whole of generation, and costs the model class one optional argument.
    """

    block_size: int
    tok_emb: nn.Embedding
    pos_emb: nn.Embedding
    drop: nn.Dropout
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
        bias: bool = False,
        gelu_approximate: GeluApproximate = "none",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dropout, bias, gelu_approximate) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None, cache: KVCache | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """idx: (batch, t) token ids -> logits (batch, t, vocab_size), and a loss if targets given.

        Positions start where the cache leaves off, so t plus whatever is already cached must not
        exceed block_size — raise ValueError when it does, rather than letting the embedding
        lookup fail somewhere deeper. With targets (batch, t), the loss is cross-entropy over
        batch and time flattened together.
        """

        _, t = idx.shape
        past = 0 if cache is None else cache.t
        if past + t > self.block_size:
            raise ValueError(
                f"Cannot forward {t} positions onto {past} already cached; "
                f"block size is only {self.block_size}"
            )
        pos = arange(past, past + t, device=idx.device)  # (t,)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (batch, t, d_model)
        for layer, block in enumerate(self.blocks):
            x = cast(Tensor, block(x, cache, layer))
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, t, vocab_size)

        loss: Tensor | None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        else:
            loss = None
        return logits, loss
