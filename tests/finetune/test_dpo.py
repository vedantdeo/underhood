"""Specification for finetune/dpo.py: a response's log-probability, then the preference loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from tests.conftest import GptFactory, TinyDims
from underhood.finetune.dpo import DpoLoss, dpo_loss, sequence_logps
from underhood.model.gpt import GPT

VOCAB, T = 7, 5


def _batch() -> tuple[Tensor, Tensor]:
    torch.manual_seed(0)
    return torch.randn(2, T, VOCAB), torch.randint(0, VOCAB, (2, T))


def test_sequence_logps_sums_each_targets_log_probability() -> None:
    """With nothing masked, it is minus the summed cross-entropy of each sequence."""
    logits, targets = _batch()
    mask = torch.ones(2, T, dtype=torch.bool)
    expected = -F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").sum(dim=1)
    assert torch.allclose(sequence_logps(logits, targets, mask), expected, atol=1e-5)


def test_the_prompt_is_not_scored() -> None:
    """Masked positions add nothing, whatever their logits say."""
    logits, targets = _batch()
    mask = torch.tensor([[False, False, True, True, True]] * 2)
    changed = logits.clone()
    changed[:, :2] = 100 * torch.randn(2, 2, VOCAB)

    kept = sequence_logps(logits, targets, mask)

    assert torch.allclose(sequence_logps(changed, targets, mask), kept, atol=1e-5)
    tail = sequence_logps(logits[:, 2:], targets[:, 2:], mask[:, 2:])
    assert torch.allclose(kept, tail, atol=1e-5)


def test_a_policy_that_has_not_moved_from_its_reference_scores_log_two() -> None:
    """Before training the policy is the reference: the margin is zero, and sigmoid(0) = 1/2."""
    chosen, rejected = torch.tensor([-4.0, -9.0]), torch.tensor([-6.0, -2.0])

    out = dpo_loss(chosen, rejected, chosen.clone(), rejected.clone(), beta=0.1)

    assert math.isclose(out.loss.item(), math.log(2), rel_tol=1e-6)
    assert torch.equal(out.chosen_rewards, torch.zeros(2))
    assert torch.equal(out.rejected_rewards, torch.zeros(2))


@pytest.mark.parametrize(
    ("margin", "beta", "loss"),
    [
        pytest.param(0.0, 0.1, math.log(2), id="no preference yet"),
        pytest.param(10 * math.log(3), 0.1, math.log(4 / 3), id="prefers the chosen response"),
        pytest.param(-10 * math.log(3), 0.1, math.log(4), id="prefers the rejected response"),
        pytest.param(math.log(3), 1.0, math.log(4 / 3), id="a tenth the margin at ten times beta"),
    ],
)
def test_the_loss_is_minus_log_sigmoid_of_beta_times_the_margin(
    margin: float, beta: float, loss: float
) -> None:
    zero = torch.zeros(1)
    out = dpo_loss(torch.tensor([margin]), zero, zero, zero, beta=beta)
    assert math.isclose(out.loss.item(), loss, rel_tol=1e-5), out.loss.item()


@pytest.mark.parametrize(
    ("shift", "moves"),
    [
        pytest.param((5.0, 5.0, 0.0, 0.0), False, id="the policy finds both responses likelier"),
        pytest.param((5.0, 0.0, 5.0, 0.0), False, id="the chosen response was likelier already"),
        pytest.param((0.0, 0.0, 5.0, 5.0), False, id="the reference finds both likelier"),
        pytest.param((5.0, 0.0, 0.0, 0.0), True, id="only the policy's preference grew"),
    ],
)
def test_only_the_difference_of_differences_moves_the_loss(
    shift: tuple[float, float, float, float], moves: bool
) -> None:
    """The reference cancels how likely a response was to begin with; only what training changed
    about the policy's preference counts."""
    chosen, rejected = torch.tensor([-3.0, -7.0]), torch.tensor([-5.0, -4.0])
    ref_chosen, ref_rejected = torch.tensor([-4.0, -6.0]), torch.tensor([-5.0, -5.0])
    d_chosen, d_rejected, d_ref_chosen, d_ref_rejected = shift

    before = dpo_loss(chosen, rejected, ref_chosen, ref_rejected).loss.item()
    after = dpo_loss(
        chosen + d_chosen,
        rejected + d_rejected,
        ref_chosen + d_ref_chosen,
        ref_rejected + d_ref_rejected,
    ).loss.item()

    assert math.isclose(before, after, rel_tol=1e-6) is not moves, (before, after)


def test_the_misranked_pair_pulls_hardest() -> None:
    """d loss / d policy_chosen is -beta * sigmoid(-beta * margin) / batch: a pair the policy
    already ranks right barely pulls, and one it ranks wrong pulls hard."""
    beta = 0.5
    chosen = torch.tensor([-3.0, -1.0], requires_grad=True)
    rejected = torch.tensor([-1.0, -3.0], requires_grad=True)
    zero = torch.zeros(2)

    dpo_loss(chosen, rejected, zero, zero, beta=beta).loss.backward()

    weight = beta * torch.sigmoid(-beta * (chosen - rejected).detach()) / 2
    assert chosen.grad is not None and rejected.grad is not None
    assert torch.allclose(chosen.grad, -weight, atol=1e-6)
    assert torch.allclose(rejected.grad, weight, atol=1e-6)
    assert chosen.grad[0].abs() > chosen.grad[1].abs(), "the first pair is the misranked one"


def test_the_rewards_are_beta_times_the_policys_drift_and_carry_no_gradient() -> None:
    chosen = torch.tensor([-2.0, -6.0], requires_grad=True)
    rejected = torch.tensor([-4.0, -3.0], requires_grad=True)
    ref_chosen, ref_rejected = torch.tensor([-3.0, -5.0]), torch.tensor([-4.0, -1.0])

    out = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.2)

    assert torch.allclose(out.chosen_rewards, 0.2 * (chosen - ref_chosen).detach())
    assert torch.allclose(out.rejected_rewards, 0.2 * (rejected - ref_rejected).detach())
    assert not out.chosen_rewards.requires_grad and not out.rejected_rewards.requires_grad


def test_the_loss_passes_a_numerical_gradient_check() -> None:
    """Autograd against finite differences, in float64, through all four inputs."""
    torch.manual_seed(0)
    inputs = tuple(torch.randn(3, dtype=torch.float64, requires_grad=True) for _ in range(4))
    assert torch.autograd.gradcheck(lambda *logps: dpo_loss(*logps, beta=0.3).loss, inputs)


def test_a_few_steps_teach_a_real_model_to_prefer_the_chosen_response(
    make_gpt: GptFactory, dims: TinyDims
) -> None:
    """End to end: a policy, its frozen copy as the reference, one prompt and two answers to it."""
    policy, reference = make_gpt(), make_gpt()
    reference.requires_grad_(False)
    prompt, chosen, rejected = [1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13]
    sequences = torch.tensor([prompt + chosen, prompt + rejected])
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, len(prompt) - 1 :] = True
    assert inputs.shape[1] <= dims.block_size

    def logps(model: GPT) -> Tensor:
        logits, _ = model(inputs)
        return sequence_logps(logits, targets, mask)

    with torch.no_grad():
        ref_chosen, ref_rejected = logps(reference)

    def scored() -> DpoLoss:
        pol_chosen, pol_rejected = logps(policy)
        return dpo_loss(pol_chosen[None], pol_rejected[None], ref_chosen[None], ref_rejected[None])

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    first = scored().loss.item()
    for _ in range(20):
        loss = scored().loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    last = scored()

    assert math.isclose(first, math.log(2), rel_tol=1e-5), "the policy starts as the reference"
    assert last.loss.item() < first - 0.1, f"loss went {first:.3f} -> {last.loss.item():.3f}"
    assert float(last.chosen_rewards - last.rejected_rewards) > 0, "the rewards rank it too"
