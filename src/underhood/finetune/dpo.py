"""DPO from scratch: how likely a model finds a response, and the loss that teaches it to prefer one
response over another without drifting far from a frozen reference.

The tests in tests/finetune/test_dpo.py are the specification: sequence_logps, then dpo_loss.

Reference: Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model is Secretly a
Reward Model", equation 7 and section 5.
"""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor

from underhood import config


def sequence_logps(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Each sequence's summed log-probability of its targets, over the positions `mask` keeps.

    logits (batch, t, vocab), targets (batch, t) token ids, mask (batch, t) bool -> (batch,).
    The mask keeps the response and drops the prompt: what the model answered is scored, not what
    it was asked.
    """
    raise NotImplementedError


class DpoLoss(NamedTuple):
    """The loss to step on, and each pair's implicit rewards for logging."""

    loss: Tensor  # scalar, meaned over the batch
    chosen_rewards: Tensor  # (batch,) beta * (policy - reference) on the chosen response, detached
    rejected_rewards: Tensor  # (batch,) the same on the rejected response


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float = config.DPO_BETA,
) -> DpoLoss:
    """-log sigmoid(beta * margin), meaned over the batch, where the margin is
    (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected).

    All four are sequence_logps, (batch,). The rewards carry no gradient.
    """
    raise NotImplementedError
