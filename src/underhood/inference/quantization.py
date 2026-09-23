"""What quantization costs and saves on a local model, measured against its own bf16 weights.

Every variant is converted here from one bf16 checkpoint, so the rows differ in bits and nothing
else. Speed comes from each variant's own greedy generations on the ten prompts in prompts.jsonl;
quality is how far its next-token distributions sit from bf16's, teacher-forced on bf16's text.

uv run underhood-quant-bench
uv run underhood-quant-bench --model mlx-community/SmolLM2-135M-Instruct   # a quick smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm.convert import convert
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

from underhood import config
from underhood.data import DATA_DIR

PROMPTS = Path(__file__).with_name("prompts.jsonl")
MODELS_DIR = DATA_DIR / "models"

# Token ids in, logits of shape (batch, length, vocab) out: an mlx_lm model, or a test's fake.
LanguageModel = Callable[[mx.array], mx.array]


class HfTokenizer(Protocol):
    """What this module uses of the HF tokenizer that TokenizerWrapper forwards to, untyped."""

    bos_token: str | None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str


@dataclass(frozen=True)
class Timed:
    """One generation: its tokens, the seconds to the first, and the decode rate after it."""

    tokens: tuple[int, ...]
    ttft_s: float
    decode_tps: float | None  # None when only one token came back


@dataclass(frozen=True)
class Reference:
    """bf16's greedy generations, and its own distributions over them, by prompt id."""

    tokens: dict[str, tuple[int, ...]]
    log_probs: dict[str, npt.NDArray[np.float32]]  # numpy, so no later variant's peak counts them


@dataclass
class Variant:
    name: str
    weight_gb: float
    peak_gb: float
    runs: dict[str, Timed]
    texts: dict[str, str]
    ttft_by_length: dict[int, float]
    kl: dict[str, float] = field(default_factory=dict[str, float])
    top1: dict[str, float] = field(default_factory=dict[str, float])
    divergence: dict[str, int | None] = field(default_factory=dict[str, int | None])


def load_prompts(path: Path = PROMPTS) -> list[Prompt]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Prompt(**json.loads(line)) for line in lines if line.strip()]


def variant_dir(repo: str, bits: int, root: Path = MODELS_DIR) -> Path:
    """Where the `bits`-bit conversion of `repo` lives, named without any -bf16 suffix."""
    name = repo.rsplit("/", 1)[-1].removesuffix("-bf16")
    return root / f"{name}-q{bits}"


def snapshot(repo: str) -> Path:
    """Every file of `repo`, in the local Hugging Face cache."""
    return Path(snapshot_download(repo))


def prepare(repo: str, bits: int, root: Path = MODELS_DIR) -> Path:
    """Quantize `repo` to `bits` once; later calls find the conversion on disk."""
    target = variant_dir(repo, bits, root)
    if not (target / "config.json").exists():
        root.mkdir(parents=True, exist_ok=True)
        convert(
            str(snapshot(repo)),  # a repo id fails on huggingface-hub 1.32: incomplete snapshot
            mlx_path=str(target),
            quantize=True,
            q_bits=bits,
            q_group_size=config.QUANT_GROUP_SIZE,
        )
        gc.collect()
        mx.clear_cache()
    return target


def chat_tokens(tokenizer: TokenizerWrapper, text: str) -> list[int]:
    """`text` as one user turn, templated and encoded the way mlx_lm's own CLI does it."""
    messages = [{"role": "user", "content": text}]
    templated = cast(
        str, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    )
    hf = cast(HfTokenizer, tokenizer)
    add_bos = hf.bos_token is None or not templated.startswith(hf.bos_token)
    return hf.encode(templated, add_special_tokens=add_bos)


def time_stream(tokens: Iterable[int], clock: Callable[[], float] = time.perf_counter) -> Timed:
    """Drain a lazy token stream, timing the first token from the start and the rest after it."""
    started = clock()
    got: list[int] = []
    first = last = started
    for token in tokens:
        last = clock()
        if not got:
            first = last
        got.append(token)
    if not got:
        raise ValueError("the stream produced no tokens")
    decode_tps = (len(got) - 1) / (last - first) if len(got) > 1 else None
    return Timed(tuple(got), first - started, decode_tps)


def generate_timed(
    model: nn.Module, tokenizer: TokenizerWrapper, prompt: Sequence[int], max_tokens: int
) -> Timed:
    """Greedy generation from `prompt`, timed; a stop token, if one came, is the last token."""
    stream = stream_generate(
        model, tokenizer, list(prompt), max_tokens=max_tokens, sampler=make_sampler(temp=0.0)
    )
    return time_stream(response.token for response in stream)


def prefill_ttft(model: nn.Module, tokenizer: TokenizerWrapper, length: int, repeats: int) -> float:
    """Median seconds to the first token from a `length`-token prompt, after one warm-up."""
    prompt = list(range(length))  # arbitrary ids: prefill time does not depend on which
    generate_timed(model, tokenizer, prompt, max_tokens=1)
    return median(
        generate_timed(model, tokenizer, prompt, max_tokens=1).ttft_s for _ in range(repeats)
    )


def log_probs(model: LanguageModel, prompt: Sequence[int], continuation: Sequence[int]) -> mx.array:
    """Row i is the log-distribution, over the vocab, that continuation token i was drawn from."""
    sequence = mx.array([*prompt, *continuation])[None]
    logits = model(sequence)[0, len(prompt) - 1 : -1].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def kl_divergence(reference: mx.array, other: mx.array) -> mx.array:
    """KL(reference || other) at each position, in nats, from two arrays of log-probabilities."""
    return (mx.exp(reference) * (reference - other)).sum(axis=-1)


def top1_agreement(reference: mx.array, other: mx.array) -> float:
    """The fraction of positions where both put their highest probability on the same token."""
    return cast(float, (reference.argmax(axis=-1) == other.argmax(axis=-1)).mean().item())


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """The index of the first token where two generations differ, or None if they are equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def weight_bytes(model: nn.Module) -> int:
    """Every parameter's storage, which for a quantized layer includes its scales and biases."""
    return sum(cast(mx.array, array).nbytes for _, array in tree_flatten(model.parameters()))


def run_variant(
    name: str, path: str, prompts: Sequence[Prompt], reference: Reference | None
) -> tuple[Variant, Reference]:
    """Measure one variant; with no reference yet, this one becomes it."""
    model, tokenizer = cast(tuple[nn.Module, TokenizerWrapper], load(path))
    encoded = {prompt.id: chat_tokens(tokenizer, prompt.text) for prompt in prompts}

    generate_timed(model, tokenizer, encoded[prompts[0].id], max_tokens=8)  # compiles kernels
    mx.reset_peak_memory()
    runs = {
        id_: generate_timed(model, tokenizer, tokens, config.QUANT_MAX_TOKENS)
        for id_, tokens in encoded.items()
    }
    variant = Variant(
        name=name,
        weight_gb=weight_bytes(model) / 1e9,
        peak_gb=mx.get_peak_memory() / 1e9,
        runs=runs,
        texts={
            id_: cast(HfTokenizer, tokenizer).decode(list(run.tokens), skip_special_tokens=True)
            for id_, run in runs.items()
        },
        ttft_by_length={
            n: prefill_ttft(model, tokenizer, n, config.QUANT_REPEATS)
            for n in config.QUANT_PREFILL_LENGTHS
        },
    )

    if reference is None:
        distributions = {
            id_: cast(npt.NDArray[np.float32], np.array(log_probs(model, tokens, runs[id_].tokens)))
            for id_, tokens in encoded.items()
        }
        return variant, Reference({id_: run.tokens for id_, run in runs.items()}, distributions)

    for id_, tokens in encoded.items():
        ours = log_probs(model, tokens, reference.tokens[id_])
        theirs = mx.array(reference.log_probs[id_])
        variant.kl[id_] = cast(float, kl_divergence(theirs, ours).mean().item())
        variant.top1[id_] = top1_agreement(theirs, ours)
        variant.divergence[id_] = first_divergence(reference.tokens[id_], runs[id_].tokens)
    return variant, reference


def _print_speed(variants: Sequence[Variant]) -> None:
    print("\nspeed and memory   (median over the ten prompts)")
    print(
        f"{'variant':>8}  {'weights':>9}  {'peak':>9}  {'TTFT':>8}  {'decode':>12}"
        f"  {'weights read':>13}"
    )
    for v in variants:
        ttft = median(run.ttft_s for run in v.runs.values())
        tps = median(run.decode_tps for run in v.runs.values() if run.decode_tps is not None)
        print(
            f"{v.name:>8}  {v.weight_gb:>6.2f} GB  {v.peak_gb:>6.2f} GB  {ttft * 1e3:>5.0f} ms"
            f"  {tps:>6.1f} tok/s  {v.weight_gb * tps:>7.1f} GB/s"
        )


def _print_prefill(variants: Sequence[Variant]) -> None:
    print(f"\ntime to first token by prompt length   (median of {config.QUANT_REPEATS})")
    print(f"{'tokens':>8}  " + "  ".join(f"{v.name:>18}" for v in variants))
    for n in config.QUANT_PREFILL_LENGTHS:
        cells = (
            f"{v.ttft_by_length[n] * 1e3:>6.0f} ms {n / v.ttft_by_length[n]:>6,.0f}/s"
            for v in variants
        )
        print(f"{n:>8}  " + "  ".join(cells))


def _print_quality(variants: Sequence[Variant], reference: Reference) -> None:
    print("\nagainst bf16   (teacher-forced on bf16's generations, weighted by token)")
    print(
        f"{'variant':>8}  {'mean KL':>12}  {'top-1 agree':>11}  {'identical':>9}"
        f"  {'first divergence':>16}"
    )
    counts = {id_: len(tokens) for id_, tokens in reference.tokens.items()}
    total = sum(counts.values())
    for v in variants:
        if not v.kl:
            continue
        kl = sum(v.kl[id_] * n for id_, n in counts.items()) / total
        top1 = sum(v.top1[id_] * n for id_, n in counts.items()) / total
        diverged = [i for i in v.divergence.values() if i is not None]
        identical = len(v.divergence) - len(diverged)
        where = f"token {median(diverged):.0f}" if diverged else "-"
        print(
            f"{v.name:>8}  {kl:>7.4f} nats  {top1:>10.1%}  {identical:>4}/{len(v.divergence):<4}"
            f"  {where:>16}"
        )


def write_outputs(
    path: Path, title: str, prompts: Sequence[Prompt], variants: Sequence[Variant]
) -> None:
    """Every variant's answer to every prompt, side by side, for reading rather than scoring."""
    lines = [f"# {title}: the ten prompts, variant by variant", ""]
    for prompt in prompts:
        lines += [f"## {prompt.id}", "", *(f"> {line}" for line in prompt.text.splitlines()), ""]
        for v in variants:
            score = f": KL {v.kl[prompt.id]:.4f}, top-1 {v.top1[prompt.id]:.1%}" if v.kl else ""
            lines += [f"**{v.name}**{score}", "", v.texts[prompt.id].strip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="bf16 against its own quantized variants")
    parser.add_argument("--model", default=config.LOCAL_MODEL, help="an unquantized MLX repo")
    repo = cast(str, parser.parse_args(argv).model)

    prompts = load_prompts()
    paths = {"bf16": str(snapshot(repo))}
    paths |= {f"q{bits}": str(prepare(repo, bits)) for bits in config.QUANT_BITS}

    variants: list[Variant] = []
    reference: Reference | None = None
    for name, path in paths.items():
        print(f"measuring {name}", flush=True)
        variant, reference = run_variant(name, path, prompts, reference)
        variants.append(variant)
        gc.collect()
        mx.clear_cache()
    assert reference is not None

    title = repo.rsplit("/", 1)[-1].removesuffix("-bf16")
    print(f"\n{title}, greedy, at most {config.QUANT_MAX_TOKENS} new tokens")
    _print_speed(variants)
    _print_prefill(variants)
    _print_quality(variants, reference)
    outputs = DATA_DIR / f"quant-{title}.md"
    write_outputs(outputs, title, prompts, variants)
    print(f"\nside by side: {outputs}")


if __name__ == "__main__":
    main()
