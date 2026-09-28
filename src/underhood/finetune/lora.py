"""LoRA from scratch: a frozen linear layer plus a trainable low-rank update, and the surgery that
fits one into a model.

The tests in tests/finetune/test_lora.py are the specification. Make them green top to bottom:
LoRALinear, then merge, then apply_lora.

Reference: Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", sections 4.1 and
7. Do not read peft's or mlx_lm's LoRA until yours passes; then read both and compare.
"""

from __future__ import annotations

from torch import Tensor, nn

from underhood import config


class LoRALinear(nn.Module):
    """`base(x) + scale * x @ A.T @ B.T`, where `base` is frozen and only A and B train.

    A is (rank, in_features) and starts random; B is (out_features, rank) and starts at zero, so
    until training moves B the layer computes exactly what `base` did. `scale` is alpha / rank.
    Keep these attribute names; the tests read them.
    """

    base: nn.Linear
    lora_a: nn.Parameter
    lora_b: nn.Parameter
    scale: float

    def __init__(
        self, base: nn.Linear, rank: int = config.LORA_RANK, alpha: float = config.LORA_ALPHA
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., in_features) -> (..., out_features)."""
        raise NotImplementedError

    def merge(self) -> nn.Linear:
        """A new plain nn.Linear computing the same function, the update folded into its weight.

        This layer is left as it was.
        """
        raise NotImplementedError


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...] = config.LORA_TARGETS,
    rank: int = config.LORA_RANK,
    alpha: float = config.LORA_ALPHA,
) -> list[str]:
    """Freeze all of `model`, then wrap each nn.Linear held under a `targets` name in a LoRALinear.

    A target is the attribute name a layer sits under (`q_proj`), wherever it is in the tree.
    Returns the dotted paths wrapped, in module order; a target that matches nothing is a
    ValueError naming it.
    """
    raise NotImplementedError
