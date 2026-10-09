"""Loss curves from underhood-train-gpu's run folders, plotted together: by step, and by wall clock.

uv run underhood-plot-curves                         # every run under data/gpu/
uv run underhood-plot-curves data/gpu/tinystories data/gpu/smoke --out a100-vs-mps.png
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from underhood import config, data

GPU_DIR = data.DATA_DIR / "gpu"


class Row(NamedTuple):
    """One evaluation: training.loop.Snapshot plus the seconds since the run began."""

    iteration: int
    train_loss: float
    val_loss: float
    seconds: float


class Curve(NamedTuple):
    name: str
    device: str
    rows: tuple[Row, ...]


def read_rows(path: Path) -> list[Row]:
    """A curve.jsonl in iteration order, one row per iteration with the last write winning.

    A resumed run restarts its clock and rewrites the rows after its checkpoint; seconds here run on
    across sessions instead.
    """
    by_iteration: dict[int, Row] = {}
    offset = previous = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        seconds = float(raw["seconds"])
        if seconds < previous:  # a new session's clock
            offset += previous
        previous = seconds
        row = Row(
            int(raw["iteration"]),
            float(raw["train_loss"]),
            float(raw["val_loss"]),
            seconds + offset,
        )
        by_iteration[row.iteration] = row
    return [by_iteration[i] for i in sorted(by_iteration)]


def load_curve(run: Path) -> Curve:
    meta = run / "run.json"
    device = json.loads(meta.read_text(encoding="utf-8"))["device"] if meta.exists() else "unknown"
    return Curve(run.name, str(device), tuple(read_rows(run / "curve.jsonl")))


def find_runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/curve.jsonl"))


def ms_per_iter(rows: Sequence[Row]) -> float | None:
    """Wall-clock ms a step from the first row to the last, evaluation included."""
    if len(rows) < 2:
        return None
    first, last = rows[0], rows[-1]
    return (last.seconds - first.seconds) / (last.iteration - first.iteration) * 1000


def summary(curve: Curve) -> str:
    last = curve.rows[-1]
    rate = ms_per_iter(curve.rows)
    pace = f", {rate:,.1f} ms/iter" if rate is not None else ""
    return f"{curve.name}: {curve.device}, {last.iteration:,} steps, val {last.val_loss:.4f}{pace}"


def plot(curves: Sequence[Curve], out: Path) -> None:
    """Two panels sharing the loss axis: by step, and by seconds on a log scale. One colour per run,
    val solid and train dashed."""
    colors = config.PLOT_SERIES_COLORS
    if not curves:
        raise ValueError("no runs to plot")
    if len(curves) > len(colors):
        raise ValueError(f"{len(curves)} runs, but only {len(colors)} colours: plot fewer at once")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (by_step, by_time) = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True, facecolor=config.PLOT_SURFACE
    )
    for ax, xlabel in ((by_step, "step"), (by_time, "seconds (log)")):
        ax.set_facecolor(config.PLOT_SURFACE)
        ax.grid(True, color=config.PLOT_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(config.PLOT_GRID)
        ax.tick_params(colors=config.PLOT_TEXT_MUTED, labelsize=9)
        ax.set_xlabel(xlabel, color=config.PLOT_TEXT_MUTED, fontsize=9)
    by_step.set_ylabel("loss", color=config.PLOT_TEXT_MUTED, fontsize=9)
    by_time.set_xscale("log")

    for curve, color in zip(curves, colors, strict=False):
        steps = [r.iteration for r in curve.rows]
        seconds = [max(r.seconds, 0.1) for r in curve.rows]  # log axis
        marker = "o" if len(curve.rows) <= config.PLOT_MARKERS_UP_TO else None
        for ax, xs in ((by_step, steps), (by_time, seconds)):
            ax.plot(
                xs,
                [r.val_loss for r in curve.rows],
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                label=curve.name if curve.device == "unknown" else f"{curve.name} ({curve.device})",
            )
            ax.plot(
                xs,
                [r.train_loss for r in curve.rows],
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label="_nolegend_",
            )
        last = curve.rows[-1]
        by_time.annotate(
            f"{last.val_loss:.2f}",
            (seconds[-1], last.val_loss),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=config.PLOT_TEXT,
        )

    handles, labels = by_step.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=min(len(labels), 4),
        frameon=False,
        fontsize=8,
        labelcolor=config.PLOT_TEXT,
        bbox_to_anchor=(0.995, 0.995),
    )
    fig.text(0.01, 0.975, "Loss by step and by wall clock", color=config.PLOT_TEXT, fontsize=11)
    fig.text(0.01, 0.925, "solid: val · dashed: train", color=config.PLOT_TEXT_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=config.PLOT_DPI, bbox_inches="tight", facecolor=config.PLOT_SURFACE)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot run folders' loss curves together.")
    parser.add_argument("runs", nargs="*", type=Path, help="run folders; default: all under --root")
    parser.add_argument("--root", type=Path, default=GPU_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: <root>/curves.png")
    args = parser.parse_args(argv)
    runs: list[Path] = args.runs or find_runs(args.root)
    curves = [load_curve(run) for run in runs]
    out: Path = args.out or args.root / "curves.png"
    plot(curves, out)
    for curve in curves:
        print(summary(curve))
    print(f"plot: {out}")


if __name__ == "__main__":
    main()
