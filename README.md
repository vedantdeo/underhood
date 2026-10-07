# underhood

The from-scratch half of the Entropic plan. Everything here is written by hand and checked against a
reference implementation: your tokenizer against tiktoken, your attention against PyTorch's, later your
GPT against its own loss curve and your LoRA against a before-and-after eval.

The plan is in `~/workspace/MLAI/ML/llm-engineer-roadmap.md` (Track B). Weekly log is the one in the
aigent repo.

## Setup

```bash
uv sync
uv run underhood-check      # picks MPS, times a matmul against CPU
uv run underhood-fetch      # downloads tiny Shakespeare into data/ (git-ignored)
uv run ptw . -x -q          # the working loop: stops on the next function to write
```

`uv sync` installs the `train` and `mlx` groups (torch, transformers, mlx-lm and the rest) by
default. The package itself depends on nothing, so another project, such as aigent, can install it
for `config` and `gpu/` alone and never pull in torch.

The specs run in dependency order, not alphabetically, so `-x` always stops on the next thing to
write rather than on whichever file sorts first. That order lives in `tests/conftest.py`.

Once a module is green, its script does something:

```bash
uv run underhood-bpe-compare  # your tokenizer's compression against tiktoken's
uv run underhood-train        # loss curve, ms/iter, and a checkpoint under data/
uv run underhood-sample       # generate from that checkpoint
uv run underhood-kv-bench     # what the cache is worth, then what batching is worth
uv run underhood-quant-bench  # Llama 3.2 3B in bf16 against its own 8-, 6- and 4-bit conversions
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

The packages are named for what the code does, not for the week it was written in; this table is
the only place the weeks appear.

| Week | Module | You write | Verified against |
|------|--------|-----------|------------------|
| 1 | `tokenizer/bpe.py` | get_stats, merge, train, encode, decode | roundtrip tests, then tiktoken via `underhood-bpe-compare` |
| 1 | `model/attention.py` | causal mask, single head, multi-head | `torch.nn.functional.scaled_dot_product_attention` |
| 2 | `model/gpt.py` | FeedForward, Block, GPT | cross-entropy at init, causality, its own logits |
| 2 | `training/loop.py` | split, batches, loss estimate, the loop itself | loss curve on MPS |
| 2 | `model/attention.py` | KVCache, and a mask that widens over cached keys | stepping through a cache matches one full forward |
| 2 | `inference/sampling.py` | temperature, top-k, top-p, generate | filter behaviour on known distributions; the same tokens with a cache and without |
| 3 | `inference/quantization.py` | bf16 against 8-, 6- and 4-bit conversions of the same checkpoint: TTFT, decode rate, memory | KL divergence from bf16's next-token distributions; the side-by-side outputs |
| 4 | `finetune/lora.py`, `finetune/dpo.py` | LoRA layer; DPO loss | toy fine-tune eval; gradient check |
| 5 | `model/pretrained.py` | GPT-2's bias and tanh-GELU switches; its weights renamed into yours | Hugging Face's logits and loss; greedy text via `underhood-gpt2-check` |
| 5 | `training/scale.py` | warmup-cosine lr, bf16 autocast, gradient accumulation, resumable checkpoints, the run | one big batch; a crashed run resumed; loss curve on an A100 vs MPS |

Rules for this repo: write the implementation before reading the reference code, then read it and
compare. Keep the tests as the spec; add a test before you extend an interface.

## GPU day runbook

Once `tests/training/test_scale.py` is green, and only after the rental is approved:

0. Locally, `uv run underhood-train-gpu --estimate`: minutes and dollars on an A100 from
   `config.GPU_USD_PER_HOUR` and a guessed MFU (`underhood.gpu.pricing`), nothing fetched or trained. Bring that number to
   the rental's go-ahead.
1. Locally, `uv run underhood-train-gpu --max-iters 3 --micro-batch 8 --name mps` for MPS's ms/iter
   on the same model (it needs `uv run underhood-fetch-tinystories` first, about 2.2 GB down).
2. Rent one A100 (RunPod or Lambda) with a PyTorch image; clone this repo, `uv sync`.
3. `uv run underhood-fetch-tinystories`, then `uv run underhood-train-gpu` inside `tmux`.
   A preempted or killed run carries on from `data/gpu/tinystories/ckpt.pt` when started again.
4. Copy back `data/gpu/tinystories/` (curve, settings, checkpoint), then stop the instance.

## Snap and roll

Twice a week both repos branch `main` and promote the branch a day later, all times IST:

| | Snap | Roll |
|---|---|---|
| first half | Mon 00:00 (Sun night) | Tue 00:00 (Mon night) |
| second half | Thu 00:00 (Wed night) | Fri 00:00 (Thu night) |

- **Snap** creates the branch `snap-YYYY-MM-DD` from `main`, and keeps the last eight (a month).
- **Roll** tags the latest snap branch's head `roll-YYYY-MM-DD` and moves the `stable` tag to it,
  only if CI passed on that commit; otherwise nothing moves and the run goes red.
- **A fix during the window** lands on `main` first, then is cherry-picked onto the snap branch
  (`git checkout snap-YYYY-MM-DD && git cherry-pick <sha> && git push`).
- `.github/workflows/release.yml` does it; run a step by hand from the Actions tab.
- aigent follows: its `main` points at underhood's snap branch during the window and at `stable`
  otherwise.

## Layout

- `src/underhood/config.py`     every tunable constant, in one place
- `src/underhood/device.py`     device selection and the matmul check
- `src/underhood/gpu/`         renting a GPU and what it costs; imports no torch, so a caller can price a run without loading it
- `src/underhood/data.py`       corpus download, and TinyStories as GPT-2 token ids
- `src/underhood/tokenizer/`    BPE, and the comparison against tiktoken
- `src/underhood/model/`        attention, the KV cache, the GPT that stacks them, and GPT-2's weights in it
- `src/underhood/training/`     batching, the training loop, and the GPU-scale run
- `src/underhood/inference/`    sampling, the benchmark that prices the KV cache, and quantization
- `src/underhood/finetune/`     LoRA, the DPO loss, and the mlx-lm toy fine-tune
- `tests/`                      the specification, one file per module, run in dependency order

## Questions parked for later

Things worth measuring on your own model once it trains, rather than arguing about in advance.

**How much does the positional path reach the logits?** With weight tying, `lm_head` is `tok_emb`
transposed, so a logit picks up a term `e_v · p_t` — the dot product of a token embedding with a
position embedding. That is a learned position-conditional prior over the vocabulary, which may be
useful signal or may be leakage. Take a trained checkpoint, compute logits normally, then again
from `h - pos_emb(t)`, and compare the two distributions: KL, or how often the top-1 token agrees.
Near-total agreement means the path carries nothing at the output; 85% means it is doing real work
and removing it would cost you.

Subtracting it in `GPT.forward` is not obviously the fix either way — `ln_f` sits between the
residual stream and `lm_head`, so subtracting before the norm gets partly undone by it, and
subtracting after fights a scale mismatch. The architectures that took this seriously removed
absolute position from the residual stream rather than taking it back out: RoPE rotates `q` and `k`
inside attention and adds nothing to the stream (`reference/roformer-rope.pdf`), and NoPE drops
positional encoding entirely, on the argument that a causal mask already implies order.

**Is random-offset sampling worth its uneven coverage?** `get_batch` draws start offsets at random
with replacement, so a token is visited a Poisson-distributed number of times — at 24 expected
visits, most land between 14 and 34, and none of it is coordinated. A partitioned sampler would fix
that: cut the training split into non-overlapping windows of `block_size`, shuffle them once per
epoch, and every token is seen exactly once per pass. Divisibility then starts to matter, and you
would truncate to a whole number of batches to drop the ragged one.

What the current sampler buys in exchange is a mild free augmentation. A token appears at a
different offset inside the window every time it is drawn, so it is predicted sometimes from three
tokens of context and sometimes from a hundred — different tasks, same data. nanoGPT and Karpathy's
lecture both sample randomly, which is why this repo does.

The experiment: same seed, same config, swap only the sampler, compare validation curves. If
partitioning wins, the augmentation was worth less than the coverage; if it loses, the reverse. One
number either way, and nobody agrees on the answer in advance.
