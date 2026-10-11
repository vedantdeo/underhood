"""Tool-use SFT: Qwen3 learns, from aigent's extract_figures conversations, to call calculate for a
scaled amount and answer in JSON. One example per assistant turn, its prompt rendered exactly as at
inference; the loss falls on that turn alone. LoRA through PEFT, in fp16 on a CUDA card.

uv run underhood-tool-sft --data data/figures --out data/figures/run          # the run
uv run underhood-tool-sft --data data/figures --out /tmp/smoke --max-steps 2  # a smoke run

The conversations come from aigent: python -m aigent.extraction.figure_training build.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from underhood import config

GENERATION = "<|im_start|>assistant\n"
IGNORE = -100
_THINK = "<think>\n\n</think>\n\n"

Message = dict[str, object]
Render = Callable[[Sequence[Message], Sequence[Message], bool], str]


@dataclass(frozen=True)
class Example:
    """One assistant turn to learn: the prompt it is generated from, and the turn itself."""

    prompt: str
    target: str


def examples(conversation: Mapping[str, object], render: Render) -> list[Example]:
    """One example per assistant turn. The prompt is the conversation so far with the generation
    prompt added, as at inference; the target is the turn as the template writes it, from the last
    assistant opening in the conversation up to that turn, less the empty think block the
    generation prompt already opened. Raises ValueError for a template with no assistant opening."""
    messages = cast(list[Message], conversation["messages"])
    tools = cast(list[Message], conversation["tools"])
    out: list[Example] = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        rendered = render(messages[: i + 1], tools, False)
        start = rendered.rfind(GENERATION)
        if start < 0:
            raise ValueError(f"turn {i} has no assistant opening: {rendered[-60:]!r}")
        turn = rendered[start + len(GENERATION) :]
        body = turn.removeprefix(_THINK).rstrip("\n")
        out.append(Example(prompt=render(messages[:i], tools, True), target=body))
    return out


@dataclass(frozen=True)
class Encoded:
    """Token ids and their labels, the prompt's masked out."""

    input_ids: list[int]
    labels: list[int]


def encode(example: Example, tokenize: Callable[[str], list[int]], limit: int) -> Encoded | None:
    """The example as ids with the prompt masked, or None when it is longer than `limit` and could
    only be trained on cut."""
    prompt, target = tokenize(example.prompt), tokenize(example.target)
    if len(prompt) + len(target) > limit:
        return None
    return Encoded(prompt + target, [IGNORE] * len(prompt) + target)


def pad(batch: Sequence[Encoded], pad_id: int) -> dict[str, Tensor]:
    """A batch right-padded to its longest example, padding masked from loss and attention."""
    width = max(len(e.input_ids) for e in batch)
    ids = [e.input_ids + [pad_id] * (width - len(e.input_ids)) for e in batch]
    labels = [e.labels + [IGNORE] * (width - len(e.labels)) for e in batch]
    mask = [[1] * len(e.input_ids) + [0] * (width - len(e.input_ids)) for e in batch]
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


@dataclass(frozen=True)
class Settings:
    """What `lora` and `train` read, defaulting to config."""

    rank: int = config.TOOL_SFT_RANK
    alpha: int = config.TOOL_SFT_ALPHA
    dropout: float = config.TOOL_SFT_DROPOUT
    targets: tuple[str, ...] = field(default=config.TOOL_SFT_TARGETS)
    epochs: int = config.TOOL_SFT_EPOCHS
    learning_rate: float = config.TOOL_SFT_LEARNING_RATE
    batch: int = config.TOOL_SFT_BATCH
    accum: int = config.TOOL_SFT_ACCUM
    warmup_ratio: float = config.TOOL_SFT_WARMUP_RATIO
    eval_steps: int = config.TOOL_SFT_EVAL_STEPS
    save_steps: int = config.TOOL_SFT_SAVE_STEPS


def lora(model: PreTrainedModel, settings: Settings) -> PeftModel:
    """The model with LoRA adapters on `settings.targets`, only they trainable. PEFT keeps them in
    float32 on an fp16 model (`autocast_adapter_dtype`), so mixed precision can unscale them."""
    adapted = cast(
        PeftModel,
        get_peft_model(
            model,
            LoraConfig(
                r=settings.rank,
                lora_alpha=settings.alpha,
                lora_dropout=settings.dropout,
                target_modules=list(settings.targets),
                task_type="CAUSAL_LM",
            ),
        ),
    )
    return adapted


def train(
    model: PreTrainedModel,
    pad_id: int,
    train_set: Sequence[Encoded],
    valid_set: Sequence[Encoded],
    out: Path,
    settings: Settings,
    max_steps: int | None = None,
) -> dict[str, float | int]:
    """LoRA-train `model`, resuming from a checkpoint under `out` when there is one, then save the
    adapter to out/adapter and the run's numbers to out/metrics.json."""
    cuda = torch.cuda.is_available()
    adapted = lora(model, settings)
    checkpoints = out / "checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoints),
        per_device_train_batch_size=settings.batch,
        per_device_eval_batch_size=settings.batch,
        gradient_accumulation_steps=settings.accum,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.epochs,
        max_steps=max_steps if max_steps is not None else -1,
        warmup_steps=settings.warmup_ratio,  # a float under 1 is a share of the steps
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=settings.eval_steps,
        save_steps=settings.save_steps,
        save_total_limit=2,
        fp16=cuda,
        gradient_checkpointing=cuda,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=adapted,
        args=args,
        train_dataset=list(train_set),
        eval_dataset=list(valid_set),
        data_collator=lambda batch: pad(batch, pad_id),
    )
    started = time.perf_counter()
    resume = checkpoints.exists() and any(checkpoints.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=resume or None)
    evaluated = trainer.evaluate()
    adapted.save_pretrained(str(out / "adapter"))
    metrics: dict[str, float | int] = {
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "eval_loss": evaluated["eval_loss"],
        "seconds": round(time.perf_counter() - started, 1),
        "train_examples": len(train_set),
        "valid_examples": len(valid_set),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def load(path: Path) -> list[dict[str, object]]:
    """The conversations in one JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach Qwen3 to call calculate for figures.")
    parser.add_argument("--data", type=Path, required=True, help="holds train.jsonl, valid.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="run folder: adapter, metrics")
    parser.add_argument("--model", default=config.TOOL_SFT_MODEL)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early, for a smoke run")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def render(messages: Sequence[Message], tools: Sequence[Message], generate: bool) -> str:
        return str(
            tokenizer.apply_chat_template(
                list(messages),
                tools=list(tools),
                tokenize=False,
                add_generation_prompt=generate,
                enable_thinking=False,
            )
        )

    def tokenize(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    sets: dict[str, list[Encoded]] = {}
    for name in ("train", "valid"):
        cut = [e for c in load(args.data / f"{name}.jsonl") for e in examples(c, render)]
        kept = [x for x in (encode(e, tokenize, config.TOOL_SFT_MAX_TOKENS) for e in cut) if x]
        sets[name] = kept
        print(f"{name}: {len(kept)} examples from {len(cut)} assistant turns")

    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if cuda else torch.float32
    )
    if cuda:
        model.enable_input_require_grads()  # checkpointing needs grads at the frozen embeddings
    args.out.mkdir(parents=True, exist_ok=True)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    metrics = train(
        model, pad_id, sets["train"], sets["valid"], args.out, Settings(), args.max_steps
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
