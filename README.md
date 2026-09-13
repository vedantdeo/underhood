# underhood

The from-scratch half of the Entropic plan. Everything here is written by hand and checked against a
reference implementation: your tokenizer against tiktoken, your attention against PyTorch's, later your
GPT against its own loss curve and your LoRA against a before-and-after eval.

The plan is in `~/workspace/MLAI/ML/llm-engineer-roadmap.md` (Track B). Weekly log is the one in the
entropic repo.

## Setup

```bash
uv sync
uv run underhood-check      # picks MPS, times a matmul against CPU
uv run underhood-fetch      # downloads tiny Shakespeare into data/ (git-ignored)
uv run pytest -q            # red until you implement; the failures are the to-do list
```

## Editor

`.vscode/` is committed, so the settings arrive with the clone; VS Code will offer the extensions on
first open (Ruff, Python, Pylance) and everything below is already wired up.

| On save | Does what |
|---------|-----------|
| `ruff format` | whitespace and layout, to the 100-char limit in `pyproject.toml` |
| `source.fixAll.ruff` | every safe autofix, including removing unused imports |
| `source.organizeImports.ruff` | sorts imports — the formatter does not do this |

`ruff.importStrategy` is `fromEnvironment`, so the editor runs `.venv/bin/ruff` rather than the copy
inside the extension. That is what keeps it reading `pyproject.toml`, and keeps the editor from
disagreeing with `uv run ruff check`. Run `uv sync` before opening the folder or there is no ruff to
find.

Two things it will not do: rename a variable that shadows a builtin (`A001` is never a safe fix), and
type-check — Pylance does that, and its warnings are signal, not noise. On another editor, point
whatever you use at the same `.venv/bin/ruff` and you get the same behaviour.

## How the weeks map

| Week | Module | You write | Verified against |
|------|--------|-----------|------------------|
| 1 | `week01/bpe.py` | get_stats, merge, train, encode, decode | roundtrip tests, then tiktoken via `underhood-bpe-compare` |
| 1 | `week01/attention.py` | causal mask, single head, multi-head | `torch.nn.functional.scaled_dot_product_attention` |
| 2 | `week02/gpt.py` | blocks, training loop on tiny Shakespeare | loss curve on MPS |
| 2 | `week02/sampling.py`, `week02/kv_cache.py` | temperature, top-p, top-k; KV cache | speedup at 512 tokens |
| 4 | `week04/lora.py`, `week04/dpo.py` | LoRA layer; DPO loss | toy fine-tune eval; gradient check |
| 5 | cloud GPU day | a 10 to 30M GPT on a rented A100 | loss curve vs the MPS run |

Rules for this repo: write the implementation before reading the reference code, then read it and
compare. Keep the tests as the spec; add a test before you extend an interface.

## Layout

- `src/underhood/device.py`   device selection and the matmul check
- `src/underhood/data.py`     corpus download
- `src/underhood/week01/`     BPE and attention skeletons
- `tests/`                    the specification, one file per module
