"""Fetch the corpora used for training from scratch.

uv run underhood-fetch               # tiny Shakespeare, for the MPS runs
uv run underhood-fetch-tinystories   # TinyStories as GPT-2 token ids, for the GPU run
"""

from __future__ import annotations

import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tiktoken
from huggingface_hub import hf_hub_download

from underhood import config

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def tiny_shakespeare() -> Path:
    """Download tiny Shakespeare (about 1 MB) once; return its path."""
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / "tinyshakespeare.txt"
    if not target.exists():
        print(f"downloading {TINY_SHAKESPEARE}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE, target)
    return target


STORY_SEPARATOR = "<|endoftext|>"
ENCODE_BATCH = 4096  # stories per tiktoken call; tiktoken threads within a call


def gpt2_vocab_size() -> int:
    """GPT-2's vocabulary, 50257: the GPU model's embedding rows."""
    return tiktoken.get_encoding(config.GPU_TOKENIZER).n_vocab


def read_ids(path: Path) -> npt.NDArray[np.uint16]:
    """A file written by tinystories(), memory-mapped rather than read."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def _stories(text_path: Path) -> Iterator[list[str]]:
    """The file's stories in batches, each stripped of the separator between them."""
    batch: list[str] = []
    lines: list[str] = []
    with text_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() == STORY_SEPARATOR:
                batch.append("".join(lines).strip())
                lines = []
                if len(batch) == ENCODE_BATCH:
                    yield batch
                    batch = []
            else:
                lines.append(line)
    if lines:
        batch.append("".join(lines).strip())
    if batch:
        yield batch


def _encode(text_path: Path, target: Path) -> None:
    """Encode every story with GPT-2's tokenizer, each followed by <|endoftext|>, as uint16."""
    encoding = tiktoken.get_encoding(config.GPU_TOKENIZER)
    started, total = time.perf_counter(), 0
    partial = target.with_suffix(".partial")
    with partial.open("wb") as out:
        for batch in _stories(text_path):
            for ids in encoding.encode_ordinary_batch([s for s in batch if s]):
                ids.append(encoding.eot_token)
                np.asarray(ids, dtype=np.uint16).tofile(out)
                total += len(ids)
    partial.rename(target)
    print(f"{target.name}: {total:,} tokens in {time.perf_counter() - started:,.0f}s")


def tinystories() -> tuple[Path, Path]:
    """Download TinyStories and encode it once; return the (train, val) id files.

    The dataset ships its own validation file, so nothing is cut from the training stories.
    """
    DATA_DIR.mkdir(exist_ok=True)
    paths: list[Path] = []
    for name in (config.GPU_TRAIN_FILE, config.GPU_VAL_FILE):
        target = DATA_DIR / "tinystories" / f"{Path(name).stem}.bin"
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            text = Path(hf_hub_download(config.GPU_DATASET, name, repo_type="dataset"))
            _encode(text, target)
        paths.append(target)
    return paths[0], paths[1]


def tinystories_main() -> None:
    for path in tinystories():
        print(f"{path}: {len(read_ids(path)):,} tokens")


def main() -> None:
    path = tiny_shakespeare()
    print(f"{path}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
