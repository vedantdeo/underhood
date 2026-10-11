"""Training at GPU scale: a learning-rate schedule, mixed precision, gradient accumulation, and a
checkpoint a preempted run can resume from exactly.

The tests in tests/training/test_train_gpu.py are the specification, in this order: lr_at,
autocast_dtype, autocast, accumulate_step, save_checkpoint and load_checkpoint, then train_run.
main() is already written: TinyStories in, a loss curve, ms/iter and a checkpoint out, and with
--estimate what the run would cost before renting anything (underhood.gpu.pricing).

uv run underhood-fetch-tinystories   # once: download and encode, about 1 GB of uint16 ids
uv run underhood-train-gpu           # on the rented A100
uv run underhood-train-gpu --max-iters 20   # the same model on MPS, for the ms/iter comparison
uv run underhood-train-gpu --estimate       # minutes and dollars on each offer, nothing trained
uv run underhood-train-gpu --dtype fp16     # force a dtype, as fp16 on an A100 against bf16
uv run underhood-kaggle push t4             # the same run on a free Kaggle T4 (gpu.kaggle)

Reference: nanoGPT's train.py. Do not read reference/ until yours passes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from underhood import config, data
from underhood.device import pick_device
from underhood.gpu.pricing import card, estimated_seconds, hourly_usd, rental_usd
from underhood.model.gpt import GPT
from underhood.training import train
from underhood.training.train import Snapshot

GPU_DIR = data.DATA_DIR / "gpu"


@dataclass(frozen=True)
class RunSettings:
    """Everything train_run reads besides the model, the data and the optimizer."""

    max_iters: int = config.GPU_MAX_ITERS
    micro_batch: int = config.GPU_MICRO_BATCH
    accum_steps: int = config.GPU_ACCUM_STEPS
    warmup_iters: int = config.GPU_WARMUP_ITERS
    max_lr: float = config.GPU_MAX_LR
    min_lr: float = config.GPU_MIN_LR
    grad_clip: float | None = config.GPU_GRAD_CLIP
    eval_interval: int = config.GPU_EVAL_INTERVAL
    eval_iters: int = config.GPU_EVAL_ITERS
    checkpoint_interval: int = config.GPU_CHECKPOINT_INTERVAL
    dtype: torch.dtype | None = None  # autocast's dtype; None trains in float32


def lr_at(iteration: int, max_iters: int, warmup_iters: int, max_lr: float, min_lr: float) -> float:
    """The learning rate for one iteration: linear warmup, then cosine decay to min_lr.

    During warmup it is max_lr * (iteration + 1) / warmup_iters, reaching max_lr on the last
    warmup step. After, a half cosine from max_lr at warmup_iters down to min_lr at max_iters,
    and min_lr from then on.
    """
    if iteration < warmup_iters:
        return max_lr * (iteration + 1) / warmup_iters
    if iteration >= max_iters:
        return min_lr
    progress = (iteration - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """bfloat16 on CUDA from Ampere (compute capability 8.0) on, float16 on older CUDA cards such
    as the T4, and None (float32) anywhere else.

    bfloat16 has float32's range, so unlike float16 it needs no GradScaler. MPS autocast exists but
    buys little for a model this size, so the MPS run stays fp32.
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device) >= (8, 0) else torch.float16


DTYPES: dict[str, torch.dtype | None] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": None,
}


def chosen_dtype(choice: str, device: torch.device) -> torch.dtype | None:
    """The autocast dtype for --dtype: "auto" defers to autocast_dtype; a DTYPES name forces it."""
    return autocast_dtype(device) if choice == "auto" else DTYPES[choice]


def autocast(device: torch.device, dtype: torch.dtype | None) -> AbstractContextManager[object]:
    """torch.autocast for device's type at dtype, or a context that does nothing when dtype is None.

    Autocast changes what the forward pass computes in, not what the weights are stored in: the
    parameters, and so their gradients, stay float32.
    """
    if dtype is None:
        return nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def accumulate_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[tuple[Tensor, Tensor]],
    grad_clip: float | None = None,
    dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """One optimizer step over several micro-batches; return their mean loss as a float.

    Clear the gradients, then for each (x, y) take the loss under autocast and backpropagate it
    divided by the number of micro-batches, so the summed gradients equal one batch of them all.
    Clip the total gradient norm to grad_clip when one is given, then step. With a scaler, as fp16
    needs, the loss is scaled before backward, unscaled before the clip, and an overflowed step is
    skipped. Raise ValueError on no micro-batches.
    """
    if not micro_batches:
        raise ValueError("accumulate_step needs at least one micro-batch")
    device = next(model.parameters()).device
    scaler = scaler or torch.amp.GradScaler(device.type, enabled=False)  # disabled: a no-op
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for x, y in micro_batches:
        with autocast(device, dtype):  # the forward only; backward reuses each op's dtype
            _, loss = model(x, y)
        assert loss is not None
        scaler.scale(loss / len(micro_batches)).backward()
        total += loss.detach().float()
    scaler.unscale_(optimizer)
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return total.item() / len(micro_batches)  # one GPU sync per step, not one per micro-batch


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    history: list[Snapshot],
    generator: torch.Generator,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Everything a resumed run needs to carry on as if never stopped, written to path.

    That is the model's and optimizer's state dicts, the number of steps taken, the curve so far,
    the generator's state, without which the resumed run draws different batches, and the fp16
    loss scale when there is a scaler. Write to a
    temporary file beside path and rename it over path, so a crash mid-write never leaves a
    half-written checkpoint where the last good one was.
    """
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "history": [tuple(row) for row in history],  # plain tuples, so weights_only can load
            "generator": generator.get_state(),
            "scaler": scaler.state_dict() if scaler is not None else {},
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, list[Snapshot]]:
    """Restore what save_checkpoint wrote into model, optimizer, generator and scaler, in place.

    A checkpoint saved without a loss scale leaves the scaler as it was. Return the number of steps
    already taken and the curve so far, as Snapshots.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])  # moments land on the parameters' device
    generator.set_state(state["generator"])
    if scaler is not None and state.get("scaler"):  # bf16 and fp32 runs, and older ones, have none
        scaler.load_state_dict(state["scaler"])
    return int(state["iteration"]), [Snapshot(*row) for row in state["history"]]


def train_run(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: Tensor,
    val_data: Tensor,
    device: torch.device,
    run: RunSettings,
    generator: torch.Generator,
    checkpoint: Path | None = None,
    on_eval: Callable[[Snapshot], None] | None = None,
) -> list[Snapshot]:
    """train.train at scale, resumable: the whole curve, including rows from before any resume.

    If checkpoint exists, load it and continue from the step it records; a finished run returns
    its curve untouched. Each iteration sets
    every param group's lr from lr_at, draws accum_steps micro-batches with train.get_batch and the
    generator, and calls accumulate_step by name (the tests stand in for it to simulate a crash).
    Evaluate as train.train does — at 0, every eval_interval and at the end, under autocast —
    passing each new row to on_eval. Save to checkpoint, when given, after every
    checkpoint_interval-th step and once more at the end. One GradScaler, enabled only under fp16,
    goes to every step and every checkpoint.
    """
    model.to(device)  # before loading, so the optimizer's moments land beside the parameters
    scaler = torch.amp.GradScaler(device.type, enabled=run.dtype == torch.float16)
    iteration, history = 0, []
    if checkpoint is not None and checkpoint.exists():
        iteration, history = load_checkpoint(checkpoint, model, optimizer, generator, scaler)
        if iteration >= run.max_iters:
            return history
    splits = {"train": train_data, "val": val_data}

    def record(at: int) -> None:
        with autocast(device, run.dtype):
            losses = train.estimate_loss(
                model, splits, device, run.micro_batch, run.eval_iters, generator
            )
        row = Snapshot(at, losses["train"], losses["val"])
        history.append(row)
        if on_eval is not None:
            on_eval(row)

    for step in range(iteration, run.max_iters):
        if step % run.eval_interval == 0:
            record(step)
        lr = lr_at(step, run.max_iters, run.warmup_iters, run.max_lr, run.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        micro_batches = [
            train.get_batch(train_data, model.block_size, run.micro_batch, device, generator)
            for _ in range(run.accum_steps)
        ]
        accumulate_step(model, optimizer, micro_batches, run.grad_clip, run.dtype, scaler)
        done = step + 1
        if checkpoint is not None and done % run.checkpoint_interval == 0 and done < run.max_iters:
            save_checkpoint(checkpoint, model, optimizer, done, history, generator, scaler)

    record(run.max_iters)
    if checkpoint is not None:
        save_checkpoint(checkpoint, model, optimizer, run.max_iters, history, generator, scaler)
    return history


def load_tokens(path: Path) -> Tensor:
    """A file of uint16 token ids as an int64 tensor, ready for get_batch."""
    return torch.from_numpy(data.read_ids(path).astype("int64"))


def _model() -> GPT:
    """The GPU-sized GPT, its head tied to the token embedding as GPT-2's is: 30M, not 49M."""
    model = GPT(
        vocab_size=data.gpt2_vocab_size(),
        block_size=config.GPU_BLOCK_SIZE,
        d_model=config.GPU_D_MODEL,
        n_heads=config.GPU_N_HEADS,
        n_layers=config.GPU_N_LAYERS,
        dropout=config.GPU_DROPOUT,
    )
    torch.nn.init.normal_(
        model.tok_emb.weight, std=config.GPU_EMBED_INIT_STD
    )  # tied, so it sets the logits' scale too
    model.lm_head.weight = model.tok_emb.weight
    return model


def _optimizer(model: GPT, device: torch.device) -> torch.optim.AdamW:
    """AdamW with decay on the matrices only: biases and LayerNorm gains are not decayed."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.GPU_WEIGHT_DECAY},
            {"params": rest, "weight_decay": 0.0},
        ],
        lr=config.GPU_MAX_LR,
        betas=config.GPU_BETAS,
        fused=device.type == "cuda",
    )


def _estimate(model: GPT, max_iters: int, chosen: str) -> str:
    """Each offer's compute time on its own card at that card's MFU, and what it charges for it
    with tax, cheapest first, `chosen` marked."""
    params = sum(p.numel() for p in model.parameters())
    tokens = max_iters * config.GPU_MICRO_BATCH * config.GPU_ACCUM_STEPS * config.GPU_BLOCK_SIZE
    rows: list[tuple[float, float, str, float]] = []  # (cost, hourly, offer, seconds)
    for offer in config.GPU_OFFERS:
        on = card(offer, config.GPU_PEAK_FLOPS)
        seconds = estimated_seconds(params, tokens, config.GPU_PEAK_FLOPS[on], config.GPU_MFU[on])
        rate = hourly_usd(offer, config.GPU_OFFERS, config.GPU_TAX)
        rows.append((rental_usd(seconds, rate), rate, offer, seconds))
    lines = [
        f"{params:,} params, {tokens:,} tokens: compute time at each card's MFU, before setup and "
        f"the download; with {config.GPU_TAX:.0%} tax:"
    ]
    for cost, rate, offer, seconds in sorted(rows):
        mark = "  <- --gpu" if offer == chosen else ""
        lines.append(f"  ${cost:.2f}  ${rate:.2f}/hr  {offer}  {seconds / 60:,.0f} min{mark}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GPU-sized GPT on TinyStories.")
    parser.add_argument("--max-iters", type=int, default=config.GPU_MAX_ITERS)
    parser.add_argument("--name", default="tinystories", help="run folder under data/gpu/")
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=config.GPU_MICRO_BATCH,
        help="smaller on MPS; accumulation rises to keep the tokens a step",
    )
    parser.add_argument(
        "--gpu", default=config.GPU_OFFER, choices=sorted(config.GPU_OFFERS), help="the rental"
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", *DTYPES],
        help="autocast's dtype; auto is bf16 from Ampere on, fp16 on older CUDA, fp32 elsewhere",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="print each offer's time and cost, then stop",
    )
    args = parser.parse_args()
    if args.estimate:
        print(_estimate(_model(), args.max_iters, args.gpu))
        return

    torch.manual_seed(config.SEED)
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_path, val_path = data.tinystories()
    train_data, val_data = load_tokens(train_path), load_tokens(val_path)

    model = _model().to(device)
    if config.GPU_COMPILE and device.type == "cuda":
        model.compile()  # in place, so state_dict keys and checkpoints stay those of a plain GPT
    optimizer = _optimizer(model, device)
    tokens_rows = config.GPU_MICRO_BATCH * config.GPU_ACCUM_STEPS
    run = RunSettings(
        max_iters=args.max_iters,
        micro_batch=args.micro_batch,
        accum_steps=max(tokens_rows // args.micro_batch, 1),
        eval_interval=min(config.GPU_EVAL_INTERVAL, args.max_iters),
        dtype=chosen_dtype(args.dtype, device),
    )
    out = GPU_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    tokens_per_step = run.micro_batch * run.accum_steps * config.GPU_BLOCK_SIZE
    print(
        f"device={device} dtype={run.dtype} params={params:,} "
        f"train={len(train_data):,} val={len(val_data):,} tokens, {tokens_per_step:,} a step"
    )

    started = time.perf_counter()
    curve = out / "curve.jsonl"
    seen: list[tuple[int, float]] = []  # (iteration, seconds) of each row this session

    def report(row: Snapshot) -> None:
        elapsed = time.perf_counter() - started
        seen.append((row.iteration, elapsed))
        print(
            f"{row.iteration:>6}  train {row.train_loss:.4f}  val {row.val_loss:.4f}  "
            f"{elapsed:,.0f}s"
        )
        with curve.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row._asdict(), "seconds": round(elapsed, 1)}) + "\n")

    generator = torch.Generator().manual_seed(config.SEED)
    train_run(
        model, optimizer, train_data, val_data, device, run, generator, out / "ckpt.pt", report
    )
    if len(seen) >= 2:
        (first, t0), (last, t1) = seen[0], seen[-1]
        per_step = (t1 - t0) / max(last - first, 1)
        print(
            f"\n{last - first} steps in {t1 - t0:,.1f}s: {per_step * 1000:,.1f} ms/iter, "
            f"{tokens_per_step / per_step:,.0f} tokens/s, evaluation included"
        )
    if device.type == "cuda":
        session = time.perf_counter() - started
        rate = hourly_usd(args.gpu, config.GPU_OFFERS, config.GPU_TAX)
        print(
            f"this session: {session / 60:,.1f} min, ${rental_usd(session, rate):.2f} on {args.gpu}"
        )
    settings = {k: str(v) for k, v in asdict(run).items()}
    (out / "run.json").write_text(json.dumps({"device": str(device), **settings}, indent=2) + "\n")
    print(f"curve: {curve}\ncheckpoint: {out / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
