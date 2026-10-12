"""Specification for finetune/tool_sft.py: examples cut one per assistant turn with the inference
prompt, the loss mask, padding, and a two-step LoRA run on a tiny Qwen3 built from a config."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import load_file
from transformers import BitsAndBytesConfig, Qwen3Config, Qwen3ForCausalLM

from underhood.finetune.tool_sft import (
    GENERATION,
    IGNORE,
    Encoded,
    Example,
    Plan,
    Settings,
    encode,
    examples,
    lora,
    model_kwargs,
    pad,
    peak_gb,
    plan,
    precision,
    prepare,
    train,
)

Message = dict[str, object]
THINK = "<think>\n\n</think>\n\n"


def qwen_like(messages: Sequence[Message], tools: Sequence[Message], generate: bool) -> str:
    """Qwen3's template in miniature: an empty think block only on the last assistant turn, and on
    the generation prompt, as Qwen3 renders them with thinking off."""
    out = f"<|im_start|>system\n{messages[0]['content']}\ntools: {json.dumps(tools)}<|im_end|>\n"
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    for i, m in enumerate(messages[1:], start=1):
        if m["role"] == "user":
            out += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        elif m["role"] == "tool":
            out += (
                f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
            )
        else:
            calls = cast(list[object], m.get("tool_calls", []))
            body = m.get("content") or "".join(
                f"<tool_call>\n{json.dumps(c)}\n</tool_call>" for c in calls
            )
            out += GENERATION + (THINK if i == last else "") + f"{body}<|im_end|>\n"
    return out + (GENERATION + THINK if generate else "")


CALL = {
    "type": "function",
    "function": {"name": "calculate", "arguments": {"expression": "2 * 10**7"}},
}
TOOL_TURN: dict[str, object] = {
    "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "passage and question"},
        {"role": "assistant", "tool_calls": [CALL]},
        {"role": "tool", "name": "calculate", "content": "20000000.0"},
        {"role": "assistant", "content": '{"figures": []}'},
    ],
    "tools": [{"type": "function", "function": {"name": "calculate"}}],
}
DIRECT: dict[str, object] = {
    "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "passage and question"},
        {"role": "assistant", "content": '{"figures": []}'},
    ],
    "tools": TOOL_TURN["tools"],
}


def test_a_tool_conversation_teaches_both_assistant_turns_from_the_inference_prompt() -> None:
    first, second = examples(TOOL_TURN, qwen_like)

    for example in (first, second):
        assert example.prompt.endswith(GENERATION + THINK), "prompted as at inference"
        assert example.target.endswith("<|im_end|>"), "taught to stop"
        assert "<think>" not in example.target
    assert first.target.startswith("<tool_call>") and "2 * 10**7" in first.target
    assert "<tool_response>\n20000000.0" in second.prompt
    assert second.target == '{"figures": []}<|im_end|>'


def test_a_conversation_with_no_tool_call_is_one_example() -> None:
    (only,) = examples(DIRECT, qwen_like)

    assert only.target == '{"figures": []}<|im_end|>'


def test_a_renderer_that_does_not_open_an_assistant_turn_is_refused() -> None:
    def broken(messages: Sequence[Message], tools: Sequence[Message], generate: bool) -> str:
        return qwen_like(messages, tools, generate).replace("assistant", "bot")

    with pytest.raises(ValueError, match="assistant"):
        examples(DIRECT, broken)


def chars(text: str) -> list[int]:
    return [ord(c) for c in text]


@pytest.mark.parametrize(
    ("limit", "kept"),
    [
        pytest.param(100, True, id="within the limit"),
        pytest.param(4, False, id="too long to train on whole"),
    ],
)
def test_encode_masks_the_prompt_and_keeps_the_target(limit: int, kept: bool) -> None:
    encoded = encode(Example(prompt="abc", target="de"), chars, limit)

    if not kept:
        assert encoded is None
        return
    assert encoded is not None
    assert encoded.input_ids == chars("abcde")
    assert encoded.labels == [IGNORE, IGNORE, IGNORE, *chars("de")]


def test_pad_right_pads_ids_masks_labels_and_marks_attention() -> None:
    batch = pad([Encoded([1, 2, 3], [IGNORE, 2, 3]), Encoded([4], [4])], pad_id=0)

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def tiny_qwen() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    return Qwen3ForCausalLM(config)


def test_lora_trains_only_the_adapters_and_keeps_them_in_float32() -> None:
    model = lora(tiny_qwen().half(), Settings())  # loaded in fp16, as on the T4

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, _ in trainable)
    assert all(p.dtype == torch.float32 for _, p in trainable), "fp16 gradients cannot be unscaled"
    for target in Settings().targets:
        assert any(target in n for n, _ in trainable), f"{target} has no adapter"


def test_train_steps_logs_finite_losses_and_saves_the_adapter(tmp_path: Path) -> None:
    rng = torch.Generator().manual_seed(1)
    data = [
        Encoded(ids := torch.randint(1, 64, (12,), generator=rng).tolist(), [IGNORE] * 6 + ids[6:])
        for _ in range(8)
    ]

    metrics = train(
        tiny_qwen(), 0, data, data[:2], tmp_path, Settings(batch=2, accum=1), max_steps=2
    )

    assert metrics["steps"] == 2
    assert math.isfinite(float(metrics["train_loss"])) and math.isfinite(
        float(metrics["eval_loss"])
    )
    saved = {p.name for p in (tmp_path / "adapter").iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors"} <= saved
    assert json.loads((tmp_path / "metrics.json").read_text())["steps"] == 2


@pytest.mark.parametrize(
    ("half", "quantize", "dtype", "four_bit"),
    [
        pytest.param(None, True, torch.float32, False, id="no CUDA: float32, never quantized"),
        pytest.param(torch.bfloat16, False, torch.bfloat16, False, id="an A30: a bf16 base"),
        pytest.param(torch.float16, True, torch.float16, True, id="a T4: an NF4 base in fp16"),
        pytest.param(torch.bfloat16, True, torch.bfloat16, True, id="QLoRA on Ampere, in bf16"),
    ],
)
def test_model_kwargs(
    half: torch.dtype | None, quantize: bool, dtype: torch.dtype, four_bit: bool
) -> None:
    kwargs = model_kwargs(half, quantize)

    assert kwargs["dtype"] == dtype
    nf4 = kwargs.get("quantization_config")
    assert (nf4 is not None) is four_bit
    if four_bit:
        assert isinstance(nf4, BitsAndBytesConfig)
        assert nf4.load_in_4bit and nf4.bnb_4bit_quant_type == "nf4"
        assert nf4.bnb_4bit_compute_dtype == half, "computes in the card's half precision"


@pytest.mark.parametrize(
    ("half", "flags"),
    [
        pytest.param(torch.bfloat16, {"bf16": True, "fp16": False}, id="bf16 from Ampere on"),
        pytest.param(torch.float16, {"bf16": False, "fp16": True}, id="fp16 on a T4"),
        pytest.param(None, {"bf16": False, "fp16": False}, id="full precision on a CPU"),
    ],
)
def test_precision(half: torch.dtype | None, flags: dict[str, bool]) -> None:
    assert precision(half) == flags


@pytest.mark.parametrize(
    ("cuda", "quantize", "checkpointing", "expected"),
    [
        pytest.param(False, True, True, Plan(False, False), id="no CUDA: neither"),
        pytest.param(True, True, True, Plan(True, True), id="a T4 by default: QLoRA, checkpointed"),
        pytest.param(True, False, True, Plan(False, True), id="an A30: bf16, checkpointed"),
        pytest.param(True, False, False, Plan(False, False), id="an A100: bf16, all activations"),
        pytest.param(True, True, False, Plan(True, False), id="QLoRA without checkpointing"),
    ],
)
def test_plan(cuda: bool, quantize: bool, checkpointing: bool, expected: Plan) -> None:
    assert plan(cuda, quantize, checkpointing) == expected


def test_a_checkpointed_run_still_trains_the_adapters(tmp_path: Path) -> None:
    model = prepare(tiny_qwen(), Plan(quantized=False, checkpointing=True))
    rng = torch.Generator().manual_seed(2)
    data = [
        Encoded(ids := torch.randint(1, 64, (12,), generator=rng).tolist(), [IGNORE] * 6 + ids[6:])
        for _ in range(4)
    ]

    train(model, 0, data, data[:1], tmp_path, Settings(batch=2, accum=1), 2, checkpointing=True)

    saved = load_file(str(tmp_path / "adapter" / "adapter_model.safetensors"))
    lora_b = [v for k, v in saved.items() if "lora_B" in k]
    assert lora_b and any(v.abs().sum() > 0 for v in lora_b), "lora_B starts at zero; it moved"


@pytest.mark.parametrize("checkpointing", [True, False], ids=["checkpointed", "not checkpointed"])
def test_a_quantized_base_is_checkpointed_only_as_planned(checkpointing: bool) -> None:
    base = tiny_qwen()
    base.__dict__["is_loaded_in_4bit"] = True  # as bitsandbytes marks it; PEFT checks for it

    model = prepare(base, Plan(quantized=True, checkpointing=checkpointing))

    assert model.is_gradient_checkpointing is checkpointing


@pytest.mark.parametrize(
    ("cuda", "expected"),
    [
        pytest.param(False, None, id="no CUDA: nothing to report"),
        pytest.param(True, 1.5, id="the most the allocator reserved, in GB"),
    ],
)
def test_peak_gb(cuda: bool, expected: float | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 3 * 2**29)

    assert peak_gb() == expected
