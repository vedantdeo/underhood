"""Specification for gpu/kaggle.py: pushing a kernel that runs this commit on a Kaggle T4, reading
its status and downloading what it kept, with git and the kaggle CLI faked."""

from __future__ import annotations

import inspect
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from underhood import config
from underhood.gpu import kaggle, kaggle_job

SHA = "0123abc"


class Cli:
    """Answers git and kaggle by the first matching argv prefix; records every call."""

    def __init__(self, answers: dict[tuple[str, ...], tuple[int, str]] | None = None) -> None:
        self.answers = {
            ("git", "rev-parse"): (0, f"{SHA}\n"),
            ("git", "status"): (0, ""),
            ("git", "branch"): (0, "  origin/vedant\n"),
            **(answers or {}),
        }
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        code, out = next(
            (
                answer
                for prefix, answer in self.answers.items()
                if tuple(argv[: len(prefix)]) == prefix
            ),
            (0, ""),
        )
        return subprocess.CompletedProcess(list(argv), code, out, "the cli said no" if code else "")

    def kaggle_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call[0] == "kaggle"]


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        pytest.param("t4", "underhood-t4", id="a short name"),
        pytest.param("tinystories-2", "underhood-tinystories-2", id="dashes and digits"),
    ],
)
def test_slug(name: str, slug: str) -> None:
    assert kaggle.slug(name) == slug


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("", id="empty"),
        pytest.param("T4", id="upper case, which Kaggle would rewrite"),
        pytest.param("t4_run", id="an underscore"),
        pytest.param("-t4", id="a leading dash"),
        pytest.param("../t4", id="a path"),
        pytest.param("t" * 41, id="longer than Kaggle keeps"),
    ],
)
def test_slug_refuses_a_name_kaggle_would_not_keep_as_is(name: str) -> None:
    with pytest.raises(ValueError, match="run name"):
        kaggle.slug(name)


def test_the_metadata_asks_for_a_private_t4_kernel_with_internet() -> None:
    meta = kaggle.metadata("someone", "t4")

    assert meta["id"] == "someone/underhood-t4"
    assert meta["title"] == "underhood-t4", "Kaggle derives the slug from the title"
    assert meta["code_file"] == kaggle.CODE_FILE
    assert (meta["language"], meta["kernel_type"]) == ("python", "script")
    assert meta["is_private"] is True
    assert (meta["enable_gpu"], meta["enable_internet"]) == (True, True)
    assert meta["machine_shape"] == config.KAGGLE_ACCELERATOR


def test_the_launcher_is_the_job_with_this_runs_arguments() -> None:
    source = kaggle.launcher(SHA, "uv run x", "data/gpu/t4")
    namespace: dict[str, object] = {"__name__": "not_main"}

    exec(compile(source, kaggle.CODE_FILE, "exec"), namespace)

    params = namespace["PARAMS"]
    assert params == {
        "repo_url": config.KAGGLE_REPO_URL,
        "sha": SHA,
        "command": "uv run x",
        "keep": "data/gpu/t4",
    }
    main = namespace["main"]
    assert callable(main) and isinstance(params, dict)
    inspect.signature(main).bind(**params)
    assert source.startswith(inspect.getsource(kaggle_job)), "the job ships unedited"
    assert source.rstrip().endswith("main(**PARAMS)")


def test_push_writes_the_kernel_and_pushes_it(tmp_path: Path) -> None:
    cli = Cli()

    folder = kaggle.push(cli, "someone", "t4", "uv run x", "data/gpu/t4", tmp_path)

    assert folder == tmp_path / "t4"
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta == kaggle.metadata("someone", "t4")
    assert (folder / kaggle.CODE_FILE).read_text() == kaggle.launcher(
        SHA, "uv run x", "data/gpu/t4"
    )
    assert cli.kaggle_calls() == [["kaggle", "kernels", "push", "-p", str(folder)]]


@pytest.mark.parametrize(
    ("answers", "match"),
    [
        pytest.param(
            {("git", "status"): (0, " M src/underhood/config.py\n")},
            "uncommitted",
            id="uncommitted changes Kaggle would not see",
        ),
        pytest.param(
            {("git", "branch"): (0, "")}, "push it first", id="a commit on no remote branch"
        ),
    ],
)
def test_push_refuses_a_commit_kaggle_cannot_check_out(
    answers: dict[tuple[str, ...], tuple[int, str]], match: str, tmp_path: Path
) -> None:
    cli = Cli(answers)

    with pytest.raises(ValueError, match=match):
        kaggle.push(cli, "someone", "t4", "uv run x", "data/gpu/t4", tmp_path)
    assert cli.kaggle_calls() == []
    assert not (tmp_path / "t4").exists()


@pytest.mark.parametrize(
    ("call", "argv"),
    [
        pytest.param(
            lambda cli, dest: kaggle.status(cli, "someone", "t4"),
            ["kaggle", "kernels", "status", "someone/underhood-t4"],
            id="status asks after the kernel",
        ),
        pytest.param(
            lambda cli, dest: kaggle.pull(cli, "someone", "t4", dest),
            ["kaggle", "kernels", "output", "someone/underhood-t4", "-p", "DEST"],
            id="pull downloads its output",
        ),
    ],
)
def test_status_and_pull_call_the_cli_for_the_kernel(
    call: Callable[[Cli, Path], object], argv: list[str], tmp_path: Path
) -> None:
    cli = Cli()

    call(cli, tmp_path)

    expected = [str(tmp_path) if arg == "DEST" else arg for arg in argv]
    assert cli.kaggle_calls() == [expected]


def test_a_failing_cli_raises_with_what_it_said() -> None:
    cli = Cli({("kaggle",): (1, "")})

    with pytest.raises(RuntimeError, match="the cli said no"):
        kaggle.status(cli, "someone", "t4")


@pytest.fixture
def kaggle_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Cli:
    cli = Cli()
    monkeypatch.setattr(kaggle, "_run", cli)
    monkeypatch.setattr(kaggle, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "KAGGLE_USER", "someone")
    return cli


def test_the_cli_push_defaults_to_the_tinystories_run(
    kaggle_cli: Cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sys.argv", ["underhood-kaggle", "push", "t4"])

    kaggle.main()

    source = (tmp_path / "kaggle" / "t4" / kaggle.CODE_FILE).read_text()
    namespace: dict[str, object] = {"__name__": "not_main"}
    exec(compile(source, kaggle.CODE_FILE, "exec"), namespace)
    params = namespace["PARAMS"]
    assert isinstance(params, dict)
    assert "underhood-fetch-tinystories" in params["command"]
    assert "underhood-train-gpu --gpu kaggle-t4 --name t4" in params["command"]
    assert params["keep"] == "data/gpu/t4"


@pytest.mark.parametrize(
    ("action", "tail"),
    [
        pytest.param("status", [], id="status names the kernel"),
        pytest.param("pull", ["-p", "kaggle/t4/output"], id="pull lands under data/kaggle"),
    ],
)
def test_the_cli_asks_after_the_runs_kernel(
    action: str, tail: list[str], kaggle_cli: Cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sys.argv", ["underhood-kaggle", action, "t4"])

    kaggle.main()

    where = [str(tmp_path / arg) if "/" in arg else arg for arg in tail]
    verb = "status" if action == "status" else "output"
    assert kaggle_cli.kaggle_calls() == [
        ["kaggle", "kernels", verb, "someone/underhood-t4", *where]
    ]


@pytest.mark.parametrize(
    ("argv", "user", "match"),
    [
        pytest.param(["status", "t4"], None, "KAGGLE_USER", id="no Kaggle user configured"),
        pytest.param(["status", "T4"], "someone", "run name", id="a name Kaggle would rewrite"),
    ],
)
def test_the_cli_exits_with_the_reason_it_refused(
    argv: list[str],
    user: str | None,
    match: str,
    kaggle_cli: Cli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "KAGGLE_USER", user)
    monkeypatch.setattr("sys.argv", ["underhood-kaggle", *argv])

    with pytest.raises(SystemExit, match=match):
        kaggle.main()
    assert kaggle_cli.kaggle_calls() == []
