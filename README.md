# underhood

LLM internals written from scratch, each checked against a reference implementation — and the
fine-tuning of an open model for a real job in its sibling agent,
[aigent](https://github.com/vedantdeo/aigent). Built over the same eight weeks, test-first: the
specs come first and run in dependency order, so the suite always stops on the next thing to build.

## Built and verified

| Module | What it is | Verified against |
|---|---|---|
| `tokenizer/bpe.py` | byte-pair encoding: training, encode, decode | round-trip tests; compression against tiktoken (`underhood-bpe-compare`) |
| `model/attention.py` | causal masking, single- and multi-head attention, a KV cache | PyTorch's `scaled_dot_product_attention`; cached decoding equals one full forward |
| `model/gpt.py` | a GPT: feed-forward, blocks, weight tying | cross-entropy at init, causality, its own loss curve |
| `model/pretrained.py` | GPT-2's weights loaded into that GPT | Hugging Face's logits and loss (`underhood-gpt2-check`) |
| `training/train.py`, `train_gpu.py` | the training loop; warmup-cosine, bf16 or scaled fp16 autocast, gradient accumulation, resumable checkpoints | loss curves on MPS, a T4 and an A100; a killed run resumed |
| `inference/sampling.py` | temperature, top-k, top-p, generation | filter behaviour on known distributions; identical tokens with and without the cache |
| `inference/quantization.py` | Llama 3.2 3B in bf16 against 8-, 6- and 4-bit | KL from bf16's next-token distributions: **q8 0.0007, q4 0.048**, at **3.4× the decode speed** for q4 |
| `finetune/lora.py`, `dpo.py` | a LoRA layer; the DPO loss | gradient checks against autograd |
| `finetune/toy.py` | a LoRA fine-tune with mlx-lm on synthetic headlines | Qwen3-1.7B (4-bit) from **11/50 to 41/50** on 50 hand-labelled real headlines |

## Tool-use fine-tune

aigent's `extract_figures` turns an annual-report passage into `{metric, value, unit, year}`
records. Scaled amounts (`₹ 1,250 crore`) are where small models fail, so **Qwen3-4B is taught to
call a calculator instead of doing the arithmetic** — trained on multi-turn conversations with real
`<tool_call>` / `<tool_response>` blocks, the loss on the assistant turns only, each prompted exactly
as at inference.

| | base Qwen3-4B | v1 | v2 (augmented data) |
|---|---|---|---|
| cases passed (43, hand-labelled) | 3 | 26 | **28** |
| valid JSON | 11 | 41 | **42** |
| called `calculate` when needed | 0/22 | 12/22 | **17/22** |
| called it when not needed | 0/21 | 1/21 | **0/21** |

bf16 LoRA (rank 16, all projections) on a rented A30, about ₹40–80 a run; evaluated on a free Kaggle
T4. Labels come from Claude through aigent, which owns every hosted-model call; this repo owns
everything from there:

- **[`tooldata/labels.py`](src/underhood/tooldata/labels.py)** refuses a label that could teach
  something false: a scale the passage never names, a call that doesn't produce the value, a negative
  the passage doesn't print as negative, a description not copied verbatim.
- **[`tooldata/augment.py`](src/underhood/tooldata/augment.py)** makes copies that stay true by
  construction — percentages rewritten consistently, years shifted, companies and currencies swapped,
  the rupee written the seven ways the reports print it — doubling the training set.
- **[`finetune/tool_sft.py`](src/underhood/finetune/tool_sft.py)** trains with PEFT and the
  Transformers `Trainer`, with QLoRA and gradient checkpointing as flags;
  **[`tool_eval.py`](src/underhood/finetune/tool_eval.py)** runs the tool loop for grading.

The most useful result was not a score: a census of v2's mistakes on 484 cases, adjudicated by hand,
found **half of them were label errors**, so the next round fixed the labels and the checks rather
than the model. The test labels were tightened at the same time, and v3 is training on the result.

## Shared tools

[`tools/`](src/underhood/tools/) holds the calculator, sandboxed file reads and writes, and the
guardrails (an injection fence, credential redaction) that aigent's agents run and the fine-tune
trains against — standard library only, so aigent installs them without pulling in torch.

## GPUs

[`gpu/`](src/underhood/gpu/) prices a run before it is rented (estimated FLOPs, a card's peak and
MFU, each provider's rate with GST), runs the pushed commit on a free Kaggle T4, and keeps run
folders on the Hugging Face Hub so a run resumes across sessions. A side study: four fp16 runs of the
same 30M-parameter model spread 0.07 in validation loss across identical seeds, so a single bf16 run
0.04 away proved nothing. Runbook: [docs/gpu.md](docs/gpu.md).

## Setup

```bash
uv sync
uv run underhood-check      # picks the device, times a matmul
uv run underhood-fetch      # tiny Shakespeare into data/ (git-ignored)
uv run pytest -q            # the specs, in dependency order
```

The package itself depends on nothing; torch, transformers and mlx-lm come from the `train` and `mlx`
dependency groups, which `uv sync` installs by default. Once a module is green its script does
something: `underhood-bpe-compare`, `underhood-train`, `underhood-sample`, `underhood-kv-bench`,
`underhood-quant-bench`, `underhood-tool-sft`, `underhood-tool-eval`, `underhood-tool-data`.

underhood is released together with aigent — snapped, checked as a pair and tagged `stable` twice a
week; see aigent's [docs/development.md](https://github.com/vedantdeo/aigent/blob/main/docs/development.md).
Questions parked for later measurement: [docs/open-questions.md](docs/open-questions.md).

## Layout

- `src/underhood/config.py` — every tunable constant
- `src/underhood/tokenizer/`, `model/`, `training/`, `inference/` — the from-scratch stack
- `src/underhood/finetune/` — LoRA, DPO, the toy fine-tune, tool-use SFT and its eval loop
- `src/underhood/tooldata/` — label checks, augmentation and the build for the tool-use fine-tune
- `src/underhood/tools/` — the tools and guardrails aigent's agents use
- `src/underhood/gpu/` — pricing, Kaggle and Hugging Face Hub runners; no torch
- `tests/` — the specification, one file per module, run in dependency order
