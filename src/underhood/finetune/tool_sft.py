"""Tool-use SFT: Qwen3 learns, from aigent's extract_figures conversations, to call calculate for a
scaled amount and answer in JSON; with --task headlines, the headline reader's tool-less rows.
One example per assistant turn, its prompt rendered exactly as at inference; the loss falls on that
turn alone. LoRA through PEFT in mixed precision, bf16 from Ampere
on and fp16 before it; with --quantize (the default) a QLoRA run on a 4-bit base, for a T4.

uv run underhood-tool-sft --data data/figures --out data/figures/run          # the run
uv run underhood-tool-sft --data data/figures --out /tmp/smoke --max-steps 2  # a smoke run
uv run underhood-tool-sft --task headlines --data data/toy-headlines --out data/headlines-4b/run

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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from underhood import config
from underhood.training.train_gpu import autocast_dtype

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
    tools = cast(list[Message], conversation.get("tools", []))
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
    keep_best: bool = False  # end on the checkpoint with the lowest eval loss, not the last


TASKS = {
    "figures": Settings(),
    "headlines": Settings(
        epochs=config.HEADLINE_SFT_EPOCHS,
        accum=config.HEADLINE_SFT_ACCUM,
        eval_steps=config.HEADLINE_SFT_EVAL_STEPS,
        save_steps=config.HEADLINE_SFT_EVAL_STEPS,
        keep_best=True,
    ),
}


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


def peak_gb() -> float | None:
    """The most GPU memory this process reserved, in GB, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_reserved() / 2**30, 2)


def train(
    model: PreTrainedModel,
    pad_id: int,
    train_set: Sequence[Encoded],
    valid_set: Sequence[Encoded],
    out: Path,
    settings: Settings,
    max_steps: int | None = None,
    half: torch.dtype | None = None,
    checkpointing: bool = False,
) -> dict[str, float | int | str]:
    """LoRA-train `model` in `half` mixed precision, resuming from a checkpoint under `out` when
    there is one, then save the adapter to out/adapter and the run's numbers to out/metrics.json."""
    adapted = lora(model, settings)
    checkpoints = out / "checkpoints"
    args = arguments(settings, checkpoints, max_steps, half, checkpointing)
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
    metrics: dict[str, float | int | str] = {
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "eval_loss": evaluated["eval_loss"],
        "seconds": round(time.perf_counter() - started, 1),
        "train_examples": len(train_set),
        "valid_examples": len(valid_set),
    }
    if settings.keep_best and trainer.state.best_model_checkpoint is not None:
        metrics["best_checkpoint"] = Path(trainer.state.best_model_checkpoint).name
    peak = peak_gb()
    if peak is not None:
        metrics["peak_gb"] = peak
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def arguments(
    settings: Settings,
    checkpoints: Path,
    max_steps: int | None = None,
    half: torch.dtype | None = None,
    checkpointing: bool = False,
) -> TrainingArguments:
    """The Trainer's arguments for a run under `settings`; keep_best saves at every evaluation and
    reloads the lowest eval loss at the end."""
    flags = precision(half)
    return TrainingArguments(
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
        save_steps=settings.eval_steps if settings.keep_best else settings.save_steps,
        save_total_limit=2,
        load_best_model_at_end=settings.keep_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=flags["bf16"],
        fp16=flags["fp16"],
        gradient_checkpointing=checkpointing,
        report_to=[],
        remove_unused_columns=False,
    )


def model_kwargs(half: torch.dtype | None, quantize: bool) -> dict[str, object]:
    """How to load the base model: in `half`, the card's mixed-precision dtype, as a 4-bit NF4
    base computing in it when `quantize`; float32 with no `half`, as on a CPU."""
    if half is None:
        return {"dtype": torch.float32}
    if not quantize:
        return {"dtype": half}
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=half,
        bnb_4bit_use_double_quant=True,
    )
    return {"dtype": half, "quantization_config": nf4}


@dataclass(frozen=True)
class Plan:
    """How a run holds the model: a 4-bit base, and activations recomputed in the backward pass."""

    quantized: bool
    checkpointing: bool


def plan(cuda: bool, quantize: bool, checkpointing: bool) -> Plan:
    """What `--quantize` and `--checkpointing` mean on this machine: neither without CUDA."""
    return Plan(quantized=cuda and quantize, checkpointing=cuda and checkpointing)


def prepare(model: PreTrainedModel, how: Plan) -> PreTrainedModel:
    """The loaded base made ready for LoRA under `how`: a 4-bit base readied for training, with
    checkpointing as planned; a half-precision base as loaded, the Trainer checkpointing it."""
    if how.quantized:
        return cast(
            PreTrainedModel,
            prepare_model_for_kbit_training(model, use_gradient_checkpointing=how.checkpointing),
        )
    return model


def precision(half: torch.dtype | None) -> dict[str, bool]:
    """The Trainer's mixed-precision flags for `half`."""
    return {"bf16": half == torch.bfloat16, "fp16": half == torch.float16}


def load(path: Path) -> list[dict[str, object]]:
    """The conversations in one JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach Qwen3 to call calculate for figures.")
    parser.add_argument("--task", choices=sorted(TASKS), default="figures", help="the settings")
    parser.add_argument("--data", type=Path, required=True, help="holds train.jsonl, valid.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="run folder: adapter, metrics")
    parser.add_argument("--model", default=config.TOOL_SFT_MODEL)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early, for a smoke run")
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=config.TOOL_SFT_QUANTIZE,
        help="a 4-bit base (QLoRA), for a card that cannot hold the base in half precision",
    )
    parser.add_argument(
        "--checkpointing",
        action=argparse.BooleanOptionalAction,
        default=config.TOOL_SFT_CHECKPOINTING,
        help="recompute activations in the backward pass: slower, far less memory",
    )
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
    half = autocast_dtype(torch.device("cuda")) if cuda else None
    how = plan(cuda, bool(args.quantize), bool(args.checkpointing))
    print(f"half precision {half}, quantized {how.quantized}, checkpointing {how.checkpointing}")
    loaded = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs(half, how.quantized))
    model = prepare(cast(PreTrainedModel, loaded), how)
    print(f"attention {model.config._attn_implementation}")
    args.out.mkdir(parents=True, exist_ok=True)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    metrics = train(
        model,
        pad_id,
        sets["train"],
        sets["valid"],
        args.out,
        TASKS[args.task],
        args.max_steps,
        half=half,
        checkpointing=how.checkpointing,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
