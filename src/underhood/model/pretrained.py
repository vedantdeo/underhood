"""GPT-2 small's weights in your own GPT: proof the architecture is the real one, not a lookalike.

The tests in tests/model/test_pretrained.py are the specification. First the two switches GPT-2
needs from your model — biases on the attention projections, and the tanh approximation of GELU —
then gpt2_state_dict, which renames and reshapes Hugging Face's weights into yours. main() is
already written: it loads the real checkpoint and compares the two models' logits and greedy text.

uv run underhood-gpt2-check

Reference: nanoGPT's GPT.from_pretrained. Do not read reference/nanogpt_model.py until yours passes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import tiktoken
import torch
from torch import Tensor
from transformers import GPT2Config, GPT2LMHeadModel

from underhood import config
from underhood.model.gpt import GPT


def gpt2_state_dict(hf_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Hugging Face's GPT2LMHeadModel state dict, renamed and reshaped into a GPT's.

    Three things differ besides the names. Hugging Face stores every projection as a Conv1D,
    whose weight is (in, out) where nn.Linear's is (out, in). It fuses q, k and v into one
    c_attn of width 3 * d_model. And it ties lm_head to the token embedding, which yours keeps
    as a separate tensor. Ignore any key that is not a weight or a bias. The result must load
    into a GPT built with bias=True and gelu_approximate="tanh" under strict=True.
    """
    raise NotImplementedError


def gpt_for(hf_config: GPT2Config) -> GPT:
    """An empty GPT shaped like a Hugging Face GPT-2 config, ready for gpt2_state_dict."""
    return GPT(
        vocab_size=hf_config.vocab_size,
        block_size=hf_config.n_positions,
        d_model=hf_config.n_embd,
        n_heads=hf_config.n_head,
        n_layers=hf_config.n_layer,
        dropout=0.0,
        bias=True,
        gelu_approximate="tanh",
    )


def _greedy(logits_of: Callable[[Tensor], Tensor], ids: Tensor, steps: int) -> list[int]:
    for _ in range(steps):
        logits = logits_of(ids)
        ids = torch.cat((ids, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    return cast(list[int], ids[0].tolist())


def main() -> None:
    print(f"loading {config.GPT2_MODEL}")
    hf = GPT2LMHeadModel.from_pretrained(config.GPT2_MODEL).eval()
    ours = gpt_for(hf.config)
    ours.load_state_dict(gpt2_state_dict(hf.state_dict()), strict=True)
    ours.eval()

    encoding = tiktoken.get_encoding(config.GPU_TOKENIZER)
    ids = torch.tensor([encoding.encode(config.GPT2_CHECK_PROMPT)])
    with torch.no_grad():
        theirs = cast(Tensor, hf(ids).logits)
        mine, _ = ours(ids)
        gap = (theirs - mine).abs().max().item()
        hf_ids = _greedy(lambda x: cast(Tensor, hf(x).logits), ids, config.GPT2_CHECK_TOKENS)
        our_ids = _greedy(lambda x: ours(x)[0], ids, config.GPT2_CHECK_TOKENS)

    params = sum(p.numel() for p in ours.parameters())
    print(
        f"params: {params:,} (untied lm_head; Hugging Face's tied count is {hf.num_parameters():,})"
    )
    print(f"largest logit gap: {gap:.2e}")
    print(f"hugging face: {encoding.decode(hf_ids)!r}")
    print(f"yours:        {encoding.decode(our_ids)!r}")
    print("greedy tokens match" if our_ids == hf_ids else "greedy tokens DIFFER")


if __name__ == "__main__":
    main()
