"""The toy fine-tune: mlx-lm's LoRA on a small local model, over aigent's template headlines.

uv run underhood-toy-lora    # trains, and writes the adapter to data/toy-headlines/adapter

Scaffolding around the exercise rather than the exercise: your own LoRA is finetune/lora.py. The
rows come from aigent, rendered as its eval will ask them, and aigent's `local-1.7b-toy`
client serves the adapter this writes:

    cd ../aigent
    uv run python -m aigent.extraction.synthetic --out ../underhood/data/toy-headlines
"""

from __future__ import annotations

import argparse
import sys
import types
from collections.abc import Sequence
from pathlib import Path

from mlx_lm import lora as mlx_lora

from underhood import config
from underhood.data import DATA_DIR

TOY_DIR = DATA_DIR / "toy-headlines"


def arguments(data: Path = TOY_DIR, adapter: Path = TOY_DIR / "adapter") -> types.SimpleNamespace:
    """mlx_lm.lora's own defaults, with this run's settings laid over them.

    mlx's `scale` is the multiplier itself, where your layer's is alpha / rank, so it is passed as
    that quotient: the same adapter under both names.
    """
    settings = dict(mlx_lora.CONFIG_DEFAULTS)
    settings.update(
        model=config.TOY_MODEL,
        train=True,
        data=str(data),
        adapter_path=str(adapter),
        iters=config.TOY_ITERS,
        batch_size=config.TOY_BATCH_SIZE,
        learning_rate=config.TOY_LEARNING_RATE,
        num_layers=config.TOY_NUM_LAYERS,
        steps_per_eval=config.TOY_STEPS_PER_EVAL,
        val_batches=-1,
        mask_prompt=True,
        seed=config.SEED,
        lora_parameters={
            "rank": config.LORA_RANK,
            "dropout": 0.0,
            "scale": config.LORA_ALPHA / config.LORA_RANK,
        },
    )
    return types.SimpleNamespace(**settings)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="underhood-toy-lora", description=__doc__)
    parser.add_argument("--data", type=Path, default=TOY_DIR, help="holds train.jsonl, valid.jsonl")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not (args.data / "train.jsonl").exists():
        raise SystemExit(
            f"no training rows in {args.data}; write them from aigent first:\n"
            "  uv run python -m aigent.extraction.synthetic --out ../underhood/data/toy-headlines"
        )
    mlx_lora.run(arguments(args.data, args.data / "adapter"))
