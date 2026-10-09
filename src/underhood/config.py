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
QUANT_BITS = (8, 6, 4)
QUANT_GROUP_SIZE = 64
QUANT_MAX_TOKENS = 256
QUANT_PREFILL_LENGTHS = (128, 512, 2048)
QUANT_REPEATS = 3

# LoRA: the update's rank, its scale as alpha / rank, the dropout on its input, and the
# projections that get one.
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "v_proj")

# DPO: how hard the loss holds the policy to its reference; smaller lets it drift further.
DPO_BETA = 0.1

# Toy fine-tune (mlx-lm) over aigent's template headlines. First guesses until a run prints its
# validation loss. The adapter takes LORA_RANK and LORA_ALPHA above, so it means what yours means.
TOY_MODEL = "mlx-community/Qwen3-1.7B-4bit"
TOY_ITERS = 200
TOY_BATCH_SIZE = 4
TOY_LEARNING_RATE = 1e-4
TOY_NUM_LAYERS = -1  # mlx-lm counts from the top; -1 adapts every layer, as apply_lora does
TOY_STEPS_PER_EVAL = 50
TOY_GRAD_CHECKPOINT = True  # recompute activations in the backward pass; same result, in 16 GB
# Projection sets the toy run compares, keyed by adapter folder name; `qv` is apply_lora's.
TOY_TARGET_SETS: dict[str, tuple[str, ...]] = {
    "all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "qv": LORA_TARGETS,
}
TOY_TARGET_SET = "qv"

# GPT-2 small, loaded into your own GPT to prove the architecture matches Hugging Face's.
GPT2_MODEL = "openai-community/gpt2"
GPT2_CHECK_PROMPT = "The quick brown fox jumps over the lazy dog, and then"
GPT2_CHECK_TOKENS = 20

# GPU run: TinyStories through GPT-2's tokenizer on a rented A100. First guesses until a run
# prints its ms/iter; one step is GPU_MICRO_BATCH * GPU_ACCUM_STEPS * GPU_BLOCK_SIZE tokens.
GPU_DATASET = "roneneldan/TinyStories"
GPU_TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
GPU_VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"
GPU_TOKENIZER = "gpt2"
GPU_BLOCK_SIZE = 256
GPU_D_MODEL = 384
GPU_N_HEADS = 6
GPU_N_LAYERS = 6
GPU_DROPOUT = 0.0
GPU_EMBED_INIT_STD = 0.02  # GPT-2's; the default of 1 gives a tied head logits in the hundreds
GPU_MICRO_BATCH = 64
GPU_ACCUM_STEPS = 8
GPU_MAX_ITERS = 5000
GPU_WARMUP_ITERS = 200
GPU_MAX_LR = 6e-4
GPU_MIN_LR = 6e-5
GPU_WEIGHT_DECAY = 0.1
GPU_BETAS = (0.9, 0.95)
GPU_GRAD_CLIP = 1.0
GPU_EVAL_INTERVAL = 250
GPU_EVAL_ITERS = 50
GPU_CHECKPOINT_INTERVAL = 500
GPU_COMPILE = True

# Renting an A100: each offer's listed $/hr before tax, priced 2026-10-07, sorted by name.
GPU_OFFERS: dict[str, float] = {
    "jarvis-a100-80gb-ondemand": 1.49,
    "jarvis-a100-80gb-spot": 0.89,  # interruptible; the run resumes from its last checkpoint
    "lambda-a100-40gb": 1.99,
    "runpod-community-a100-80gb": 1.19,
    "runpod-secure-a100-80gb": 1.59,
}
GPU_OFFER = "jarvis-a100-80gb-spot"
GPU_TAX = 0.18  # GST, which India levies on each of these
GPU_PEAK_FLOPS = 312e12  # dense bf16
GPU_MFU = 0.3  # share of the peak a 30M model reaches; a guess until a run prints its ms/iter

# Loss-curve plots (training.curves). Categorical slots in assignment order, which is the data: do
# not sort. A ninth run is refused rather than given a generated hue.
PLOT_SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
PLOT_SURFACE = "#fcfcfb"
PLOT_TEXT = "#0b0b0b"
PLOT_TEXT_MUTED = "#52514e"
PLOT_GRID = "#e6e5e1"
PLOT_DPI = 150
PLOT_MARKERS_UP_TO = 20  # a curve with this many rows or fewer also gets a marker at each row
