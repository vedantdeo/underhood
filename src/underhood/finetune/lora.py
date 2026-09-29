"""LoRA from scratch: a frozen linear layer plus a trainable low-rank update, and the surgery that
fits one into a model.

The tests in tests/finetune/test_lora.py are the specification. Make them green top to bottom:
LoRALinear, then merge, then apply_lora.

Reference: Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", sections 4.1 and
7. Do not read peft's or mlx_lm's LoRA until yours passes; then read both and compare.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from underhood import config


class LoRALinear(nn.Module):
    """`base(x) + scale * dropout(x) @ A.T @ B.T`, where `base` is frozen and only A and B train.

    A is (rank, in_features) and starts random; B is (out_features, rank) and starts at zero, so
    until training moves B the layer computes exactly what `base` did. `scale` is alpha / rank.
    A and B are float32 whatever the base's dtype; dropout touches only the update's input.
    Keep these attribute names; the tests read them.
    """

    base: nn.Linear
    lora_a: nn.Parameter
    lora_b: nn.Parameter
    scale: float
    dropout: nn.Dropout

    def __init__(
        self,
        base: nn.Linear,
        rank: int = config.LORA_RANK,
        alpha: float = config.LORA_ALPHA,
        dropout: float = config.LORA_DROPOUT,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        weight = base.weight
        self.lora_a = nn.Parameter(
            nn.init.kaiming_uniform_(
                weight.new_empty(rank, base.in_features, dtype=torch.float32), a=math.sqrt(5)
            )
        )
        self.lora_b = nn.Parameter(weight.new_zeros(base.out_features, rank, dtype=torch.float32))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., in_features) -> (..., out_features), in x's dtype."""
        update = self.dropout(x).to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + (self.scale * update).to(x.dtype)

    def merge(self) -> nn.Linear:
        """A new plain nn.Linear computing the same function, the update folded into its weight.

        This layer is left as it was.
        """
        base = self.base
        merged = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        with torch.no_grad():
            folded = base.weight.float() + self.scale * self.lora_b @ self.lora_a
            merged.weight.copy_(folded.to(base.weight.dtype))
            if base.bias is not None:
                merged.bias.copy_(base.bias)
        return merged


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...] = config.LORA_TARGETS,
    rank: int = config.LORA_RANK,
    alpha: float = config.LORA_ALPHA,
    dropout: float = config.LORA_DROPOUT,
) -> list[str]:
    """Freeze all of `model`, then wrap each nn.Linear held under a `targets` name in a LoRALinear.

    A target is the attribute name a layer sits under (`q_proj`), wherever it is in the tree.
    Returns the dotted paths wrapped, in module order; a target that matches nothing is a
    ValueError naming it.
    """
    model.requires_grad_(False)

    wrapped: dict[str, nn.Linear] = {}
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear) and path.rsplit(".", 1)[-1] in targets:
            wrapped[path] = module

    found = {path.rsplit(".", 1)[-1] for path in wrapped}
    if missing := [target for target in targets if target not in found]:
        raise ValueError(f"no nn.Linear under {', '.join(missing)}")

    for path, module in wrapped.items():
        parent, _, attribute = path.rpartition(".")
        setattr(model.get_submodule(parent), attribute, LoRALinear(module, rank, alpha, dropout))

    return list(wrapped)
