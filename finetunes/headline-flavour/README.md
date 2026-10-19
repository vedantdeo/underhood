# headline-flavour — parked

The headline reader: one Indian market results headline in, one record out,
`{company, metric, quarter, direction, change_pct}`. **Parked on 2026-10-19** (Vedant): as a capstone
specialist it cost more per call than the generalist doing the job, since a headline is a dozen
answer tokens and served answer tokens ($26/M) cost more than Sonnet's ($10/M). It was taken out
of aigent's analyst in aigent's commit after `1d34890`; this folder keeps everything needed to use
it again.

## The adapter

| | |
|---|---|
| weights | Hub, private: `vistack/headline-flavour` (`adapter_model.safetensors`, sha256 `50a21910…a0f7`, 132 MB) |
| base | `Qwen/Qwen3-4B`, NF4 4-bit while training (QLoRA); answered on the fp16 base when served |
| LoRA | r 16, alpha 32, dropout 0.05, all seven projections (`adapter_config.json`, PEFT 0.21.2) |
| trained | 2026-10-13, Kaggle T4 (`kaggle-kernel.json`), 11 min, 9.8 GB peak; free |
| code | underhood at **`25e17c1`** ("Train the headline reader on the PEFT trainer"), pinned in `kaggle-run.py` |
| command | `uv run underhood-tool-sft --task headlines --data <rows> --out data/headlines-4b/run` |
| settings at that commit | `TOOL_SFT_*` with the `HEADLINE_SFT_*` overrides: 2 epochs, lr 2e-4, batch 1 × accum 4 (×2 GPUs = 8), warmup 3%, eval every 15 steps, max 2048 tokens; loss on the assistant turn only |
| run | 46 steps, train loss 0.085, eval loss 0.044 → 0.0053, best checkpoint the last (`metrics.json`) |
| score | **38/50** on aigent's 50 hand-labelled headlines (base 4B 17/50; the Week 4 MLX 1.7B 41/50), graded by `aigent.extraction.headline_eval`; reports `headlines-20261013-0627.md` and `-0627-2.md` in aigent's `evals/reports/` |

## The exact prompt

`rows/` holds the rows it was trained, validated and tested on (180 / 20 / 50), as rendered by
aigent's `extraction.synthetic` on the Qwen wire: a system message (the extraction rules and the
field list), a user message `headline: <the headline>`, and the assistant's JSON record. **A
request must match them byte for byte**: the system message in any row, and the user turn in that
form. aigent builds it with `extraction.headlines.headline_request` on `VARIANTS["zero_shot"]`,
answered with thinking off (`enable_thinking: False`). `predictions/` holds the tuned and base
models' replies to the 50 test rows.

| file | sha256 |
|---|---|
| `rows/train.jsonl` | `4b17cf9f…7396` |
| `rows/valid.jsonl` | `cf4f06e4…3220` |
| `rows/test.jsonl` | `13205969…97a7` |

## To use it again

1. **Serve:** `hosting/specialists.json` still names it, so the Modal app loads it as
   `headline-flavour` beside `report-analyst`; the adapter costs nothing until a request names it.
2. **In aigent:** bring back `hostllm-headline` / `vllm-headline` (`cfg_hostllm`), the
   `headline-flavour` price row, `served.headline_reader`, `specialists.check_headline` and
   `experts.headline_expert` with its `BUILDERS` row: `git show 1d34890 -- src/aigent/analyst
   src/aigent/llm` has them all, with their tests.
3. **Known open questions** (parked 2026-10-13): train/serve skew (learned on NF4, answered on fp16)
   and undertraining (46 steps against Week 4's 200, eval loss still falling).
