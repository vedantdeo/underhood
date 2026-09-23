"""Tunable constants for the from-scratch models.

Everything here is a knob: change a value and you get a different result. Locations and derived
values stay beside the code that computes them (see data.DATA_DIR), because those are not settings.

The model defaults are sized for an M-series MPS box, not for an A100. Karpathy's lecture uses
BLOCK_SIZE 256 / D_MODEL 384 / N_LAYERS 6; that is roughly 4x this and takes a GPU to be pleasant.
These numbers are guesses until a real run has printed its ms/iter — `uv run underhood-train` does.
"""

from __future__ import annotations

# Tokenizer: your BPE, trained on the training split only.
VOCAB_SIZE = 512

# Model
BLOCK_SIZE = 128
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.2

# Training
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
VAL_FRACTION = 0.1
SEED = 1337

# Generation
GENERATE_TOKENS = 400
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.95

# KV-cache benchmark: the roadmap's 512 tokens, with block headroom so the cache never crops.
KV_BENCH_BLOCK_SIZE = 1024
KV_BENCH_TOKENS = 512
KV_BENCH_REPEATS = 3

# Batch sweep, ascending: about granularity rather than length, so the sequences are shorter.
KV_BENCH_BATCHES = (1, 16, 64, 256)
KV_BENCH_BATCH_TOKENS = 64

# Local models (mlx-lm): one bf16 checkpoint, and every quantized variant converted from it here.
LOCAL_MODEL = "mlx-community/Llama-3.2-3B-Instruct-bf16"
QUANT_BITS = (8, 4)
QUANT_GROUP_SIZE = 64
QUANT_MAX_TOKENS = 256
QUANT_PREFILL_LENGTHS = (128, 512, 2048)
QUANT_REPEATS = 3
