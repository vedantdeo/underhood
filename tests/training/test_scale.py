"""Specification for training/scale.py: the schedule, mixed precision, accumulation, checkpoints,
then the resumable run that uses them all."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from torch import Tensor

import underhood.training.scale as scale
from tests.conftest import GptFactory, TinyDims
from underhood.model.gpt import GPT
from underhood.training.loop import Snapshot
from underhood.training.scale import (
    RunSettings,
    accumulate_step,
    autocast,
    autocast_dtype,
    load_checkpoint,
    lr_at,
    save_checkpoint,
    train_run,
)

CPU = torch.device("cpu")
MAX_ITERS, WARMUP, MAX_LR, MIN_LR = 110, 10, 1.0, 0.1


@pytest.fixture
def stream(dims: TinyDims) -> Tensor:
    return torch.tensor([i % dims.vocab_size for i in range(1024)], dtype=torch.long)


def _batch(dims: TinyDims, rows: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, dims.vocab_size, (rows, dims.block_size + 1), generator=generator)
    return ids[:, :-1], ids[:, 1:]


def _params(model: GPT) -> Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        pytest.param(0, 0.1, id="the first warmup step is a tenth of the way up"),
        pytest.param(4, 0.5, id="halfway through warmup"),
        pytest.param(9, 1.0, id="the last warmup step reaches the peak"),
        pytest.param(10, 1.0, id="the cosine starts from the peak"),
        pytest.param(35, 0.1 + 0.45 * (1 + math.sqrt(0.5)), id="a quarter in, a cosine not a line"),
        pytest.param(60, 0.55, id="halfway through the decay is halfway between peak and floor"),
        pytest.param(110, 0.1, id="the floor at max_iters"),
        pytest.param(500, 0.1, id="the floor after max_iters"),
    ],
)
def test_lr_at(iteration: int, expected: float) -> None:
    got = lr_at(iteration, MAX_ITERS, WARMUP, MAX_LR, MIN_LR)
    assert math.isclose(got, expected, abs_tol=1e-9), got


def test_lr_at_never_rises_after_warmup() -> None:
    rates = [lr_at(i, MAX_ITERS, WARMUP, MAX_LR, MIN_LR) for i in range(WARMUP, MAX_ITERS + 5)]
    assert all(a >= b for a, b in zip(rates, rates[1:], strict=False)), rates


def test_lr_at_without_warmup_starts_at_the_peak() -> None:
    assert math.isclose(lr_at(0, MAX_ITERS, 0, MAX_LR, MIN_LR), MAX_LR, abs_tol=1e-9)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        pytest.param("cuda", torch.bfloat16, id="bf16 on CUDA"),
        pytest.param("mps", None, id="fp32 on MPS"),
        pytest.param("cpu", None, id="fp32 on CPU"),
    ],
)
def test_autocast_dtype(device: str, dtype: torch.dtype | None) -> None:
    assert autocast_dtype(torch.device(device)) == dtype


@pytest.mark.parametrize(
    ("dtype", "computed_in"),
    [
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16 autocast computes a matmul in bf16"),
        pytest.param(None, torch.float32, id="no autocast leaves it in fp32"),
    ],
)
def test_autocast_changes_the_compute_not_the_weights(
    dtype: torch.dtype | None, computed_in: torch.dtype
) -> None:
    layer = torch.nn.Linear(8, 8)
    with autocast(CPU, dtype):
        out = layer(torch.randn(2, 8))
    out.float().sum().backward()
    assert out.dtype == computed_in
    assert layer.weight.dtype == torch.float32
    assert layer.weight.grad is not None and layer.weight.grad.dtype == torch.float32


@pytest.mark.parametrize(
    "split", [pytest.param(2, id="two micro-batches"), pytest.param(4, id="four micro-batches")]
)
def test_accumulating_micro_batches_steps_like_one_big_batch(
    make_gpt: GptFactory, dims: TinyDims, split: int
) -> None:
    x, y = _batch(dims, 8)
    whole, pieces = make_gpt().train(), make_gpt().train()
    accumulate_step(whole, torch.optim.SGD(whole.parameters(), lr=0.5), [(x, y)])
    chunks = list(zip(x.chunk(split), y.chunk(split), strict=True))
    accumulate_step(pieces, torch.optim.SGD(pieces.parameters(), lr=0.5), chunks)
    gap = (_params(whole) - _params(pieces)).abs().max().item()
    assert torch.allclose(_params(whole), _params(pieces), atol=1e-6), f"largest gap {gap:.2e}"


def test_accumulate_step_returns_the_mean_loss_before_the_step(
    make_gpt: GptFactory, dims: TinyDims
) -> None:
    model = make_gpt().train()
    batches = [_batch(dims, 2, seed) for seed in range(3)]
    with torch.no_grad():
        expected = sum(float(model(x, y)[1]) for x, y in batches) / 3
    loss = accumulate_step(model, torch.optim.SGD(model.parameters(), lr=0.5), batches)
    assert isinstance(loss, float)
    assert math.isclose(loss, expected, rel_tol=1e-5), (loss, expected)


def test_accumulate_step_ignores_gradients_left_from_before(
    make_gpt: GptFactory, dims: TinyDims
) -> None:
    batch = [_batch(dims, 4)]
    clean, stale = make_gpt().train(), make_gpt().train()
    for p in stale.parameters():
        p.grad = torch.full_like(p, 100.0)
    accumulate_step(clean, torch.optim.SGD(clean.parameters(), lr=0.5), batch)
    accumulate_step(stale, torch.optim.SGD(stale.parameters(), lr=0.5), batch)
    assert torch.allclose(_params(clean), _params(stale), atol=1e-6)


@pytest.mark.parametrize(
    ("grad_clip", "clipped"),
    [
        pytest.param(1e-3, True, id="a tight clip bounds the step"),
        pytest.param(None, False, id="no clip leaves the step as large as the gradient"),
    ],
)
def test_accumulate_step_clips_the_total_gradient_norm(
    make_gpt: GptFactory, dims: TinyDims, grad_clip: float | None, clipped: bool
) -> None:
    model = make_gpt().train()
    before = _params(model)
    accumulate_step(
        model, torch.optim.SGD(model.parameters(), lr=1.0), [_batch(dims, 4)], grad_clip
    )
    moved = (_params(model) - before).norm().item()
    assert (moved <= 1e-3 * (1 + 1e-4)) is clipped, moved


def test_accumulate_step_trains_under_bf16_and_keeps_fp32_weights(
    make_gpt: GptFactory, dims: TinyDims
) -> None:
    model = make_gpt().train()
    loss = accumulate_step(
        model, torch.optim.SGD(model.parameters(), lr=0.5), [_batch(dims, 4)], dtype=torch.bfloat16
    )
    assert math.isfinite(loss)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_accumulate_step_refuses_no_micro_batches(make_gpt: GptFactory) -> None:
    model = make_gpt()
    with pytest.raises(ValueError):
        accumulate_step(model, torch.optim.SGD(model.parameters(), lr=0.5), [])


def test_a_checkpoint_restores_everything_a_run_needs(
    make_gpt: GptFactory, dims: TinyDims, tmp_path: Path
) -> None:
    model = make_gpt().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    accumulate_step(model, optimizer, [_batch(dims, 4)])  # so AdamW has moments to save
    generator = torch.Generator().manual_seed(7)
    torch.randint(0, 100, (3,), generator=generator)
    history = [Snapshot(0, 3.1, 3.2), Snapshot(5, 2.5, 2.7)]
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, 5, history, generator)
    next_draw = torch.randint(0, 100, (5,), generator=generator)

    fresh = make_gpt()
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-2)
    fresh_generator = torch.Generator().manual_seed(0)
    iteration, restored = load_checkpoint(path, fresh, fresh_optimizer, fresh_generator)

    assert iteration == 5
    assert restored == history and all(isinstance(row, Snapshot) for row in restored)
    assert torch.equal(_params(fresh), _params(model))
    first = next(iter(model.parameters()))
    fresh_first = next(iter(fresh.parameters()))
    assert torch.equal(
        fresh_optimizer.state[fresh_first]["exp_avg"], optimizer.state[first]["exp_avg"]
    ), "the optimizer's moments were not restored"
    assert torch.equal(torch.randint(0, 100, (5,), generator=fresh_generator), next_draw), (
        "the generator would draw different batches after a resume"
    )
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"], "a temporary file was left behind"


def test_saving_again_replaces_the_checkpoint(
    make_gpt: GptFactory, dims: TinyDims, tmp_path: Path
) -> None:
    model = make_gpt()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    generator = torch.Generator().manual_seed(0)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 5, [], generator)
    save_checkpoint(path, model, optimizer, 10, [], generator)
    iteration, _ = load_checkpoint(path, make_gpt(), torch.optim.SGD(model.parameters()), generator)
    assert iteration == 10


SETTINGS = RunSettings(
    max_iters=10,
    micro_batch=4,
    accum_steps=2,
    warmup_iters=2,
    max_lr=1e-2,
    min_lr=1e-3,
    grad_clip=1.0,
    eval_interval=5,
    eval_iters=2,
    checkpoint_interval=5,
)


def _run(
    make_gpt: GptFactory, stream: Tensor, checkpoint: Path | None, seed: int = 0
) -> tuple[GPT, torch.optim.Optimizer, list[Snapshot], list[Snapshot]]:
    model = make_gpt().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS.max_lr)
    seen: list[Snapshot] = []
    generator = torch.Generator().manual_seed(seed)
    history = train_run(
        model, optimizer, stream, stream, CPU, SETTINGS, generator, checkpoint, seen.append
    )
    return model, optimizer, history, seen


def test_train_run_records_the_curve_and_reports_each_row(
    make_gpt: GptFactory, stream: Tensor
) -> None:
    model, optimizer, history, seen = _run(make_gpt, stream, None)
    assert [row.iteration for row in history] == [0, 5, 10]
    assert seen == history
    assert history[-1].train_loss < history[0].train_loss, history
    assert model.training, "train_run must leave the model in training mode"
    lr = optimizer.param_groups[0]["lr"]
    assert math.isclose(lr, lr_at(9, 10, 2, 1e-2, 1e-3), rel_tol=1e-9), lr


def test_train_run_saves_a_checkpoint_at_the_end(
    make_gpt: GptFactory, stream: Tensor, tmp_path: Path
) -> None:
    path = tmp_path / "ckpt.pt"
    model, _, history, _ = _run(make_gpt, stream, path)
    fresh = make_gpt()
    iteration, restored = load_checkpoint(
        path, fresh, torch.optim.AdamW(fresh.parameters()), torch.Generator()
    )
    assert iteration == 10
    assert restored == history
    assert torch.equal(_params(fresh), _params(model))


def test_a_run_that_crashes_resumes_to_exactly_where_it_would_have_been(
    make_gpt: GptFactory, stream: Tensor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash on the seventh step, after the checkpoint at five; the resumed run must not notice."""
    straight, _, straight_history, _ = _run(make_gpt, stream, None)

    path = tmp_path / "ckpt.pt"
    real_step = scale.accumulate_step
    calls = 0

    def crash_on_seventh(
        model: GPT,
        optimizer: torch.optim.Optimizer,
        micro_batches: Sequence[tuple[Tensor, Tensor]],
        grad_clip: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> float:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("the instance was preempted")
        return real_step(model, optimizer, micro_batches, grad_clip, dtype)

    monkeypatch.setattr(scale, "accumulate_step", crash_on_seventh)
    with pytest.raises(RuntimeError, match="preempted"):
        _run(make_gpt, stream, path)
    monkeypatch.setattr(scale, "accumulate_step", real_step)

    resumed, _, resumed_history, seen = _run(make_gpt, stream, path, seed=123)

    gap = (_params(straight) - _params(resumed)).abs().max().item()
    assert torch.allclose(_params(straight), _params(resumed), atol=1e-6), f"gap {gap:.2e}"
    assert resumed_history == straight_history
    assert [row.iteration for row in seen] == [5, 10], "only new rows are reported after a resume"


def test_relaunching_a_finished_run_changes_nothing(
    make_gpt: GptFactory, stream: Tensor, tmp_path: Path
) -> None:
    path = tmp_path / "ckpt.pt"
    _, _, first, _ = _run(make_gpt, stream, path)
    _, _, again, seen = _run(make_gpt, stream, path)
    assert again == first, "a finished run was evaluated again"
    assert seen == []
