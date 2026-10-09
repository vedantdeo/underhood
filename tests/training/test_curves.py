"""Spec for training.curves: reading a run folder's curve.jsonl back, and plotting runs together."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from underhood import config
from underhood.training.curves import (
    Curve,
    Row,
    find_runs,
    load_curve,
    main,
    ms_per_iter,
    plot,
    read_rows,
    summary,
)


def write_curve(path: Path, rows: list[tuple[int, float, float]]) -> None:
    """rows are (iteration, val_loss, seconds); train_loss is val_loss + 0.1."""
    path.write_text(
        "".join(
            json.dumps({"iteration": i, "train_loss": v + 0.1, "val_loss": v, "seconds": s}) + "\n"
            for i, v, s in rows
        ),
        encoding="utf-8",
    )


def make_run(
    root: Path, name: str, rows: list[tuple[int, float, float]], device: str | None
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    write_curve(run / "curve.jsonl", rows)
    if device is not None:
        (run / "run.json").write_text(json.dumps({"device": device}), encoding="utf-8")
    return run


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        pytest.param(
            [(0, 10.9, 2.7), (250, 4.0, 80.0)],
            [(0, 10.9, 2.7), (250, 4.0, 80.0)],
            id="one session reads back as written",
        ),
        pytest.param(
            [
                (0, 10.9, 3.0),
                (250, 4.0, 80.0),
                (500, 3.0, 160.0),
                (750, 2.5, 240.0),
                (750, 2.4, 30.0),
                (1000, 2.2, 110.0),
            ],
            [
                (0, 10.9, 3.0),
                (250, 4.0, 80.0),
                (500, 3.0, 160.0),
                (750, 2.4, 270.0),
                (1000, 2.2, 350.0),
            ],
            id="a resume keeps the rewritten row and carries its clock on",
        ),
    ],
)
def test_read_rows(
    tmp_path: Path,
    written: list[tuple[int, float, float]],
    expected: list[tuple[int, float, float]],
) -> None:
    path = tmp_path / "curve.jsonl"
    write_curve(path, written)
    got = [(r.iteration, r.val_loss, r.seconds) for r in read_rows(path)]
    assert got == pytest.approx(expected), got


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda", "cuda"), pytest.param(None, "unknown", id="a run with no run.json")],
)
def test_load_curve_names_the_run_and_its_device(
    tmp_path: Path, device: str | None, expected: str
) -> None:
    run = make_run(tmp_path, "tinystories", [(0, 10.9, 2.7)], device)
    curve = load_curve(run)
    assert (curve.name, curve.device, len(curve.rows)) == ("tinystories", expected, 1), curve


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param([(0, 10.9, 3.0), (200, 5.8, 64.4)], 307.0, id="two rows"),
        pytest.param(
            [(0, 10.9, 3.0), (100, 6.0, 33.0), (200, 5.8, 64.4)], 307.0, id="first to last"
        ),
        pytest.param([(0, 10.9, 3.0)], None, id="one row has no rate"),
    ],
)
def test_ms_per_iter(rows: list[tuple[int, float, float]], expected: float | None) -> None:
    got = ms_per_iter([Row(i, v + 0.1, v, s) for i, v, s in rows])
    assert got == pytest.approx(expected), got


def test_summary_states_device_steps_final_loss_and_rate() -> None:
    curve = Curve("probe-200", "cuda", (Row(0, 11.0, 10.9, 3.0), Row(200, 5.9, 5.8107, 64.4)))
    line = summary(curve)
    for part in ("probe-200", "cuda", "200 steps", "val 5.8107", "307.0 ms/iter"):
        assert part in line, line


def test_find_runs_returns_folders_with_a_curve_sorted(tmp_path: Path) -> None:
    make_run(tmp_path, "smoke", [(0, 10.9, 9.2)], "mps")
    (tmp_path / "empty").mkdir()
    make_run(tmp_path, "a100", [(0, 10.9, 2.7)], "cuda")
    assert [p.name for p in find_runs(tmp_path)] == ["a100", "smoke"]


def test_plot_writes_a_png(tmp_path: Path) -> None:
    curves = [
        Curve("a100", "cuda", (Row(0, 11.0, 10.9, 2.7), Row(250, 4.1, 4.0, 80.0))),
        Curve("smoke", "mps", (Row(0, 11.0, 10.9, 9.2), Row(20, 10.1, 10.0, 373.8))),
    ]
    out = tmp_path / "curves.png"
    plot(curves, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    ("count", "message"),
    [
        pytest.param(0, "no runs", id="no runs"),
        pytest.param(len(config.PLOT_SERIES_COLORS) + 1, "colours", id="more runs than colours"),
    ],
)
def test_plot_refuses(tmp_path: Path, count: int, message: str) -> None:
    curves = [Curve(f"r{i}", "cuda", (Row(0, 11.0, 10.9, 1.0),)) for i in range(count)]
    with pytest.raises(ValueError, match=message):
        plot(curves, tmp_path / "curves.png")
    assert not (tmp_path / "curves.png").exists()


def test_main_plots_the_given_runs_and_prints_a_line_each(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a100 = make_run(tmp_path, "a100", [(0, 10.9, 2.7), (250, 4.0, 80.0)], "cuda")
    smoke = make_run(tmp_path, "smoke", [(0, 10.9, 9.2), (20, 10.0, 373.8)], "mps")
    out = tmp_path / "plot.png"
    main([str(a100), str(smoke), "--out", str(out)])
    printed = capsys.readouterr().out
    assert out.exists() and "a100" in printed and "smoke" in printed and str(out) in printed, (
        printed
    )


def test_main_with_no_runs_finds_them_under_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(tmp_path, "a100", [(0, 10.9, 2.7), (250, 4.0, 80.0)], "cuda")
    main(["--root", str(tmp_path)])
    assert (tmp_path / "curves.png").exists() and "a100" in capsys.readouterr().out
