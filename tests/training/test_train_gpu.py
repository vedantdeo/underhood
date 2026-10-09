"""Specification for training/train_gpu.py: the schedule, mixed precision, accumulation,
checkpoints, then the resumable run that uses them all."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

import underhood.training.train_gpu as train_gpu
from tests.conftest import GptFactory, TinyDims
from underhood import config, data
from underhood.gpu.pricing import card, estimated_seconds
from underhood.model.gpt import GPT
from underhood.training.train import Snapshot
from underhood.training.train_gpu import (
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
    ("device", "capability", "dtype"),
    [
        pytest.param("cuda", (8, 0), torch.bfloat16, id="bf16 on an A100"),
        pytest.param("cuda", (9, 0), torch.bfloat16, id="bf16 on an H100"),
        pytest.param("cuda", (7, 5), torch.float16, id="fp16 on a T4, which has no bf16"),
        pytest.param("mps", None, None, id="fp32 on MPS"),
        pytest.param("cpu", None, None, id="fp32 on CPU"),
    ],
)
def test_autocast_dtype(
    device: str,
    capability: tuple[int, int] | None,
    dtype: torch.dtype | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def capability_of(_: torch.device) -> tuple[int, int]:
        if capability is None:
            raise AssertionError("only a CUDA device has a compute capability to ask for")
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_of)
    assert autocast_dtype(torch.device(device)) == dtype


@pytest.mark.parametrize(
    ("dtype", "computed_in"),
    [
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16 autocast computes a matmul in bf16"),
        pytest.param(torch.float16, torch.float16, id="fp16 autocast computes a matmul in fp16"),
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


def test_fp16_with_a_scaler_steps_like_fp32(make_gpt: GptFactory, dims: TinyDims) -> None:
    """The loss is scaled up for the backward pass and the gradients unscaled before the clip."""
    batch = [_batch(dims, 4)]
    steps = []
    for dtype in (None, torch.float16):
        model = make_gpt().train()
        before = _params(model)
        scaler = torch.amp.GradScaler("cpu", enabled=dtype is not None)
        accumulate_step(
            model, torch.optim.SGD(model.parameters(), lr=0.5), batch, 1.0, dtype, scaler
        )
        steps.append(_params(model) - before)
    fp32, fp16 = steps
    gap = ((fp16 - fp32).norm() / fp32.norm()).item()
    assert gap < 1e-2, f"the fp16 step is {gap:.1%} away from the fp32 one"


def _overflowing(make_gpt: GptFactory) -> GPT:
    """A loss near 1e4, which times GradScaler's starting scale of 65536 overflows fp16."""
    model = make_gpt().train()
    with torch.no_grad():
        model.lm_head.weight.mul_(1e4)
    return model


def test_a_scaler_skips_a_step_whose_gradients_overflowed(
    make_gpt: GptFactory, dims: TinyDims
) -> None:
    model = _overflowing(make_gpt)
    before = _params(model)
    scaler = torch.amp.GradScaler("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    accumulate_step(model, optimizer, [_batch(dims, 4)], 1.0, torch.float16, scaler)

    assert torch.equal(_params(model), before), "an overflowed step reached the weights"
    assert scaler.get_scale() == 32768.0, "the scale did not back off for the next step"


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


@pytest.mark.parametrize(
    ("overflow", "saved_with_scaler", "scale"),
    [
        pytest.param(True, True, 32768.0, id="the scale a run backed off to survives a resume"),
        pytest.param(False, False, 65536.0, id="a bf16 run's checkpoint leaves the scale at start"),
    ],
)
def test_a_checkpoint_restores_the_loss_scale(
    make_gpt: GptFactory,
    dims: TinyDims,
    tmp_path: Path,
    overflow: bool,
    saved_with_scaler: bool,
    scale: float,
) -> None:
    model = _overflowing(make_gpt) if overflow else make_gpt().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    scaler = torch.amp.GradScaler("cpu")
    accumulate_step(model, optimizer, [_batch(dims, 4)], 1.0, torch.float16, scaler)
    path, generator = tmp_path / "ckpt.pt", torch.Generator()
    save_checkpoint(path, model, optimizer, 1, [], generator, scaler if saved_with_scaler else None)

    fresh = torch.amp.GradScaler("cpu")
    load_checkpoint(path, make_gpt(), torch.optim.SGD(model.parameters()), generator, fresh)

    assert fresh.get_scale() == scale


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
    real_step = train_gpu.accumulate_step
    calls = 0

    def crash_on_seventh(
        model: GPT,
        optimizer: torch.optim.Optimizer,
        micro_batches: Sequence[tuple[Tensor, Tensor]],
        grad_clip: float | None = None,
        dtype: torch.dtype | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> float:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("the instance was preempted")
        return real_step(model, optimizer, micro_batches, grad_clip, dtype, scaler)

    monkeypatch.setattr(train_gpu, "accumulate_step", crash_on_seventh)
    with pytest.raises(RuntimeError, match="preempted"):
        _run(make_gpt, stream, path)
    monkeypatch.setattr(train_gpu, "accumulate_step", real_step)

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


@pytest.mark.parametrize(
    ("dtype", "scaled"),
    [
        pytest.param(torch.float16, True, id="fp16 scales the loss"),
        pytest.param(torch.bfloat16, False, id="bf16 has fp32's range and needs no scaling"),
        pytest.param(None, False, id="fp32 needs no scaling"),
    ],
)
def test_train_run_scales_the_loss_only_under_fp16(
    make_gpt: GptFactory,
    stream: Tensor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype | None,
    scaled: bool,
) -> None:
    stepped: list[torch.amp.GradScaler | None] = []
    saved: list[torch.amp.GradScaler | None] = []
    real_step, real_save = train_gpu.accumulate_step, train_gpu.save_checkpoint

    def spy_step(
        model: GPT,
        optimizer: torch.optim.Optimizer,
        micro_batches: Sequence[tuple[Tensor, Tensor]],
        grad_clip: float | None = None,
        dtype: torch.dtype | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> float:
        stepped.append(scaler)
        return real_step(model, optimizer, micro_batches, grad_clip, dtype, scaler)

    def spy_save(
        path: Path,
        model: GPT,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        history: list[Snapshot],
        generator: torch.Generator,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        saved.append(scaler)
        real_save(path, model, optimizer, iteration, history, generator, scaler)

    monkeypatch.setattr(train_gpu, "accumulate_step", spy_step)
    monkeypatch.setattr(train_gpu, "save_checkpoint", spy_save)
    model = make_gpt().train()
    run = RunSettings(**{**asdict(SETTINGS), "dtype": dtype})
    history = train_run(
        model,
        torch.optim.AdamW(model.parameters(), lr=run.max_lr),
        stream,
        stream,
        CPU,
        run,
        torch.Generator().manual_seed(0),
        tmp_path / "ckpt.pt",
    )

    assert len(stepped) == run.max_iters and saved, (len(stepped), len(saved))
    used = {id(s): s for s in stepped + saved}
    enabled = [s is not None and s.is_enabled() for s in used.values()]
    assert enabled == [scaled] * len(used), "one scaler, for every step and every checkpoint"
    assert history[-1].train_loss < history[0].train_loss, history


# --- the GPU run's model, optimizer and data ----------------------------------------------------

GPT2_VOCAB = 50257


@pytest.fixture
def gpu_model(monkeypatch: pytest.MonkeyPatch) -> GPT:
    monkeypatch.setattr(data, "gpt2_vocab_size", lambda: GPT2_VOCAB)  # tiktoken may download
    torch.manual_seed(0)
    return train_gpu._model()


def test_the_gpu_model_ties_its_head_to_the_embedding(gpu_model: GPT) -> None:
    assert gpu_model.lm_head.weight is gpu_model.tok_emb.weight, "one tensor, trained once"
    params = sum(p.numel() for p in gpu_model.parameters())
    untied = params + GPT2_VOCAB * config.GPU_D_MODEL
    assert 29e6 < params < 31e6 and 48e6 < untied < 50e6, f"{params:,} tied, {untied:,} untied"


def test_the_gpu_model_starts_its_embedding_at_gpt2s_scale(gpu_model: GPT) -> None:
    std = gpu_model.tok_emb.weight.std().item()
    assert abs(std - config.GPU_EMBED_INIT_STD) < 0.1 * config.GPU_EMBED_INIT_STD, std


def test_the_optimizer_decays_the_matrices_and_nothing_else(make_gpt: GptFactory) -> None:
    model = make_gpt()
    optimizer = train_gpu._optimizer(model, CPU)

    decayed, kept = optimizer.param_groups
    matrices = {id(p) for p in model.parameters() if p.dim() >= 2}
    assert {id(p) for p in decayed["params"]} == matrices
    assert {id(p) for p in kept["params"]} == {id(p) for p in model.parameters()} - matrices
    assert kept["params"], "biases and LayerNorm gains exist, and are kept out of the decay"
    assert (decayed["weight_decay"], kept["weight_decay"]) == (config.GPU_WEIGHT_DECAY, 0.0)
    assert (decayed["lr"], decayed["betas"]) == (config.GPU_MAX_LR, config.GPU_BETAS)
    assert not decayed["fused"], "fused AdamW is CUDA-only"


def test_a_token_file_loads_as_int64_ids(tmp_path: Path) -> None:
    ids = [0, 1, 40000, GPT2_VOCAB - 1]  # past 32,767, where a signed 16-bit read goes negative
    path = tmp_path / "train.bin"
    np.asarray(ids, dtype=np.uint16).tofile(path)

    tokens = train_gpu.load_tokens(path)

    assert tokens.dtype == torch.int64
    assert tokens.tolist() == ids


@pytest.mark.parametrize(
    ("flags", "chosen"),
    [
        pytest.param([], config.GPU_OFFER, id="the default offer"),
        pytest.param(
            ["--gpu", "runpod-community-a100-80gb"], "runpod-community-a100-80gb", id="one picked"
        ),
        pytest.param(["--gpu", "kaggle-t4"], "kaggle-t4", id="the free T4"),
    ],
)
def test_estimate_prices_every_offer_without_fetching_or_training(
    flags: list[str],
    chosen: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_download() -> tuple[Path, Path]:
        raise AssertionError("an estimate must not fetch TinyStories")

    monkeypatch.setattr(data, "gpt2_vocab_size", lambda: GPT2_VOCAB)
    monkeypatch.setattr(data, "tinystories", no_download)
    argv = ["underhood-train-gpu", "--estimate", "--max-iters", "100", *flags]
    monkeypatch.setattr("sys.argv", argv)

    train_gpu.main()

    header, *rows = capsys.readouterr().out.strip().splitlines()
    tokens = 100 * config.GPU_MICRO_BATCH * config.GPU_ACCUM_STEPS * config.GPU_BLOCK_SIZE
    assert f"{tokens:,} tokens" in header and f"{config.GPU_TAX:.0%} tax" in header, header
    params = sum(p.numel() for p in train_gpu._model().parameters())
    for row in rows:
        on = card(row.split()[2], config.GPU_PEAK_FLOPS)
        seconds = estimated_seconds(params, tokens, config.GPU_PEAK_FLOPS[on], config.GPU_MFU[on])
        assert f"{seconds / 60:,.0f} min" in row, f"{row} is not timed on its own card, the {on}"
    named = [row.split()[2] for row in rows]
    assert sorted(named) == sorted(config.GPU_OFFERS), "a row per offer"
    costs = [float(row.split()[0].lstrip("$")) for row in rows]
    assert costs == sorted(costs), "cheapest first"
    assert [row for row in rows if row.endswith("<- --gpu")] == [r for r in rows if chosen in r]
    rate = config.GPU_OFFERS[chosen] * (1 + config.GPU_TAX)
    assert f"${rate:.2f}/hr  {chosen}" in "\n".join(rows)
