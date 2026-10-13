"""Tool-loop inference for aigent's figure eval: each exported case answered by Qwen3, base or with
a LoRA adapter, its calculate calls run through underhood.tools and their results shown to it, up to
a few rounds. Writes the predictions aigent grades.

uv run underhood-tool-eval --test data/figures/test.jsonl --out data/figures/eval-base.jsonl
uv run underhood-tool-eval --test data/figures/test.jsonl --adapter data/figures/a30/adapter \\
    --out data/figures/eval-tuned.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from underhood import config
from underhood.tools.calc import calculate
from underhood.training.train_gpu import autocast_dtype

Message = dict[str, object]
Generate = Callable[[Sequence[Message], Sequence[Message]], str]

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class _Generates(Protocol):
    """The one model method the loop calls, which the transformers stubs do not show."""

    def generate(self, **kwargs: object) -> Tensor: ...


def tool_calls(text: str) -> list[dict[str, object]]:
    """Every well-formed <tool_call> JSON object in a reply, in order, skipping malformed ones."""
    calls: list[dict[str, object]] = []
    for body in _CALL.findall(text):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            calls.append(cast(dict[str, object], parsed))
    return calls


def _run(call: Mapping[str, object]) -> tuple[str | None, str]:
    """The expression a call made, if any, and what the tool says back."""
    arguments = call.get("arguments")
    expression = arguments.get("expression") if isinstance(arguments, dict) else None
    if call.get("name") != "calculate":
        return None, f"Error: no tool named {call.get('name')!r}"
    if not isinstance(expression, str):
        return None, "Error: 'expression' must be a string"
    try:
        return expression, str(calculate(expression))
    except ValueError as error:
        return expression, f"Error: {error}"


def answer(row: Mapping[str, object], generate: Generate, rounds: int) -> dict[str, object]:
    """One case through the loop: generate, run any calls and show their results, and stop at the
    first reply with no call. A model still calling after `rounds` replies has no answer."""
    messages = list(cast(list[Message], row["messages"]))
    tools = cast(list[Message], row.get("tools", []))
    made: list[str] = []
    for _ in range(rounds):
        reply = generate(messages, tools)
        calls = tool_calls(reply)
        if not calls:
            return {"id": row["id"], "final": reply, "calls": made, "error": None}
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": c} for c in calls],
            }
        )
        for c in calls:
            expression, said = _run(c)
            if expression is not None:
                made.append(expression)
            messages.append({"role": "tool", "name": "calculate", "content": said})
    return {
        "id": row["id"],
        "final": None,
        "calls": made,
        "error": f"still calling after {rounds} rounds",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer the figure test set through the tool loop."
    )
    parser.add_argument("--test", type=Path, required=True, help="aigent's exported test.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="where the predictions go")
    parser.add_argument("--model", default=config.TOOL_SFT_MODEL)
    parser.add_argument("--adapter", type=Path, default=None, help="a LoRA adapter to apply")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

    cuda = torch.cuda.is_available()
    half = autocast_dtype(torch.device("cuda")) if cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if cuda else "cpu"
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(args.model, dtype=half, device_map=device),
    )
    if args.adapter is not None:
        model = cast(PreTrainedModel, PeftModel.from_pretrained(model, str(args.adapter)))
    model.eval()
    generator = cast(_Generates, model)

    def generate(messages: Sequence[Message], tools: Sequence[Message]) -> str:
        ids = tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools),
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            out = generator.generate(
                **ids, max_new_tokens=config.TOOL_EVAL_MAX_NEW_TOKENS, do_sample=False
            )
        reply = tokenizer.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=False)
        return reply.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    rows = [json.loads(x) for x in args.test.read_text().splitlines() if x.strip()]
    started = time.perf_counter()
    with args.out.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            out.write(json.dumps(answer(row, generate, config.TOOL_EVAL_ROUNDS)) + "\n")
            print(f"{i}/{len(rows)} {time.perf_counter() - started:.0f}s", flush=True)
    print(f"predictions: {args.out}")


if __name__ == "__main__":
    main()
