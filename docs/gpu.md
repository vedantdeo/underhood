# Renting a GPU

Once `tests/training/test_train_gpu.py` is green:

0. Locally, `uv run underhood-train-gpu --estimate`: minutes on an A100 at a guessed MFU and what
   each offer in `config.GPU_OFFERS` charges for them with GST (`underhood.gpu.pricing`), nothing
   fetched or trained. The default is Jarvis spot; pick another with `--gpu`. Bring the number to
   the rental's go-ahead.
1. Locally, `uv run underhood-train-gpu --max-iters 3 --micro-batch 8 --name mps` for MPS's ms/iter
   on the same model (it needs `uv run underhood-fetch-tinystories` first, about 2.2 GB down).
2. Rent one A100 (RunPod or Lambda) with a PyTorch image; clone this repo, `uv sync`.
3. `uv run underhood-fetch-tinystories`, then `uv run underhood-train-gpu` inside `tmux`.
   A preempted or killed run carries on from `data/gpu/tinystories/ckpt.pt` when started again.
4. Copy back `data/gpu/tinystories/` (curve, settings, checkpoint), then stop the instance.

## Free compute

`uv run underhood-kaggle push <name> --command '...'` runs the pushed commit on a free Kaggle T4
(about 30 GPU-hours a week) and `underhood-kaggle pull <name>` brings its output back; it refuses
uncommitted or unpushed code, since the kernel clones the exact commit from GitHub. `underhood-hub`
keeps a run folder in a private Hugging Face repo so a run can resume across 12-hour sessions.
