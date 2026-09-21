"""Sampling: temperature, top-k, top-p, and the generation loop that uses them.

The tests in tests/inference/test_sampling.py are the specification, in this order:
apply_temperature, top_k_filter, top_p_filter, sample_next, then generate. main() is written for
you — it loads the checkpoint `underhood-train` wrote and prints GENERATE_TOKENS of Shakespeare.

uv run underhood-sample
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from underhood import config
from underhood.device import pick_device
from underhood.model.attention import KVCache
from underhood.model.gpt import GPT
from underhood.tokenizer.bpe import BPETokenizer
from underhood.training.loop import CHECKPOINT

Forward = tuple[Tensor, Tensor | None]


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Divide the logits by temperature: below 1 sharpens the distribution, above 1 flattens it.

    Raise ValueError for temperature <= 0. Zero temperature means argmax, which you get here by
    asking for top_k=1 instead — dividing by zero is not the way to say it.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}; use top_k=1 for argmax")
    return logits / temperature


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the k largest logits per row and set the rest to -inf, so softmax gives them no mass.

    logits: (batch, vocab). A k at or above vocab size leaves the row alone; k < 1 is a ValueError.
    """
    if k < 1:
        raise ValueError(f"top_k must be at least 1, got {k}")
    kth = logits.topk(min(k, logits.size(-1)), dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus: keep the smallest set of tokens whose probabilities reach p, the rest to -inf.

    logits: (batch, vocab). Always keep at least the single most likely token, however small p is;
    p outside (0, 1] is a ValueError. Unlike top-k, the number kept varies with how peaked the row
    is — which is the whole point of it.
    """
    if not 0 < p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    ordered, index = logits.sort(dim=-1, descending=True)
    probs = ordered.softmax(dim=-1)
    # compare the total *before* each token, so the one that crosses p is kept and the rest are not
    keep = probs.cumsum(dim=-1) - probs < p
    return logits.scatter(-1, index, ordered.masked_fill(~keep, float("-inf")))


def sample_next(
    logits: Tensor,
    temperature: float = config.TEMPERATURE,
    top_k: int | None = config.TOP_K,
    top_p: float | None = config.TOP_P,
    generator: torch.Generator | None = None,
) -> Tensor:
    """One token per row: temperature, then top-k, then top-p, then a draw from the softmax.

    logits: (batch, vocab) -> (batch, 1) of token ids. Either filter is skipped when None. Pass
    `generator` to make a draw reproducible.
    """
    scaled = apply_temperature(logits, temperature)
    if top_k is not None:
        scaled = top_k_filter(scaled, top_k)
    if top_p is not None:
        scaled = top_p_filter(scaled, top_p)
    return torch.multinomial(scaled.softmax(dim=-1), num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: GPT,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = config.TEMPERATURE,
    top_k: int | None = config.TOP_K,
    top_p: float | None = config.TOP_P,
    generator: torch.Generator | None = None,
    cache: KVCache | None = None,
) -> Tensor:
    """Extend idx (batch, t) by max_new_tokens, one at a time, returning (batch, t + max_new).

    Without a cache every step re-reads the whole context, cropped to the model's last block_size
    positions, because the position embeddings have nothing to say about a longer one. With one,
    the prompt goes through as a single prefill pass and each step after feeds only the token it
    just sampled — same tokens, far less arithmetic, but a cache cannot be cropped, so the whole
    sequence has to fit one block. Gradients are off here; this is inference.
    """
    feed = idx  # the first pass sees the whole prompt either way
    for _ in range(max_new_tokens):
        if cache is None:
            feed = idx[:, -model.block_size :]
        logits, _ = cast(Forward, model(feed, cache=cache))
        next_id = sample_next(logits[:, -1], temperature, top_k, top_p, generator)
        idx = torch.cat((idx, next_id), dim=1)
        feed = next_id
    return idx


def _load(device: torch.device) -> tuple[BPETokenizer, GPT]:
    """Rebuild the tokenizer and model that `underhood-train` checkpointed."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"no checkpoint at {CHECKPOINT}; run `uv run underhood-train` first"
        )
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    tokenizer = BPETokenizer()
    tokenizer.merges = saved["merges"]
    tokenizer.vocab = saved["vocab"]

    model = GPT(vocab_size=saved["vocab_size"], block_size=saved["block_size"])
    model.load_state_dict(saved["model"])
    return tokenizer, model.to(device).eval()


def main() -> None:
    torch.manual_seed(config.SEED)
    device = pick_device()
    tokenizer, model = _load(device)

    prompt = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
    out = generate(model, prompt, config.GENERATE_TOKENS)
    print(
        f"temperature={config.TEMPERATURE} top_k={config.TOP_K} top_p={config.TOP_P}"
        f"  {config.GENERATE_TOKENS} tokens\n"
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
