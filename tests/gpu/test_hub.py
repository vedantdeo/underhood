"""Specification for gpu/hub.py: a run folder under data/ goes up to a private Hub repo and comes
back, against a fake Hub that keeps files in memory."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

import underhood.gpu
from underhood import config, data
from underhood.gpu import hub

REPO = f"someone/{config.HUB_RUNS_REPO}"


class FakeHub:
    """Repos as {path in repo: bytes}; records what was created."""

    def __init__(self, repos: dict[str, dict[str, bytes]] | None = None) -> None:
        self.repos = repos if repos is not None else {}
        self.created: list[tuple[str, bool]] = []

    def whoami(self) -> Mapping[str, object]:
        return {"name": "someone"}

    def create_repo(self, repo_id: str, *, private: bool, exist_ok: bool) -> object:
        assert exist_ok, "a second push must not fail on the repo the first one made"
        self.created.append((repo_id, private))
        self.repos.setdefault(repo_id, {})
        return repo_id

    def upload_folder(self, *, repo_id: str, folder_path: Path, path_in_repo: str) -> object:
        for file in folder_path.rglob("*"):
            if file.is_file():
                name = f"{path_in_repo}/{file.relative_to(folder_path).as_posix()}"
                self.repos[repo_id][name] = file.read_bytes()
        return repo_id

    def list_repo_files(self, repo_id: str) -> list[str]:
        if repo_id not in self.repos:
            request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
            raise RepositoryNotFoundError(repo_id, response=httpx.Response(404, request=request))
        return list(self.repos[repo_id])

    def snapshot_download(self, repo_id: str, *, allow_patterns: str, local_dir: Path) -> object:
        for name, content in self.repos[repo_id].items():
            if fnmatch.fnmatch(name, allow_patterns):
                target = local_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        return str(local_dir)


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "gpu" / "t4"
    (folder / "nested").mkdir(parents=True)
    (folder / "curve.jsonl").write_text('{"iteration": 0}\n')
    (folder / "nested" / "ckpt.pt").write_bytes(b"\x00weights")
    return folder


def _contents(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def test_the_data_dir_is_the_one_underhood_data_writes_to() -> None:
    assert underhood.gpu.DATA_DIR == data.DATA_DIR


def test_a_pushed_run_folder_pulls_back_byte_for_byte(run_folder: Path, tmp_path: Path) -> None:
    fake = FakeHub()
    kept = _contents(run_folder)

    where = hub.push(fake, run_folder, tmp_path)
    for file in run_folder.rglob("*"):
        if file.is_file():
            file.unlink()
    found = hub.pull(fake, run_folder, tmp_path)

    assert fake.created == [(REPO, True)], "the runs repo is created private"
    assert where == f"{REPO}/gpu/t4"
    assert found is True
    assert _contents(run_folder) == kept


@pytest.mark.parametrize(
    ("repos", "why"),
    [
        pytest.param({}, "no runs repo yet", id="the runs repo does not exist"),
        pytest.param({REPO: {"gpu/a100/ckpt.pt": b"x"}}, "another run", id="only another run"),
        pytest.param(
            {REPO: {"gpu/t4-long/ckpt.pt": b"x"}},
            "a run whose name starts with this one's",
            id="a run named with this one's as a prefix",
        ),
    ],
)
def test_pull_says_so_when_the_hub_has_no_such_run(
    repos: dict[str, dict[str, bytes]], why: str, tmp_path: Path
) -> None:
    folder = tmp_path / "gpu" / "t4"

    assert hub.pull(FakeHub(repos), folder, tmp_path) is False, why
    assert not folder.exists(), "nothing was written for a run the Hub does not have"


@pytest.mark.parametrize(
    ("action", "relative", "match"),
    [
        pytest.param("push", "gpu/missing", "no folder", id="pushing a folder that is not there"),
        pytest.param("push", "gpu/empty", "empty", id="pushing an empty folder"),
        pytest.param("push", "../elsewhere", "inside", id="pushing a folder outside data/"),
        pytest.param("pull", "../elsewhere", "inside", id="pulling into a folder outside data/"),
    ],
)
def test_hub_refuses_a_folder_it_should_not_touch(
    action: str, relative: str, match: str, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "gpu" / "empty").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "secret.txt").write_text("not a run")
    fake = FakeHub({REPO: {}})
    folder = data_dir / relative

    with pytest.raises(ValueError, match=match):
        hub.push(fake, folder, data_dir) if action == "push" else hub.pull(fake, folder, data_dir)
    assert fake.repos == {REPO: {}}, "a refused call reached the Hub"


@pytest.fixture
def hub_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[FakeHub, Path]:
    fake = FakeHub()
    monkeypatch.setattr(hub, "_api", lambda: fake)
    monkeypatch.setattr(hub, "DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    return fake, tmp_path


def test_the_cli_pushes_and_pulls_a_folder_named_from_the_working_directory(
    hub_cli: tuple[FakeHub, Path],
    run_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, data_dir = hub_cli
    named = str(run_folder.relative_to(data_dir.parent))

    for argv in (["push", named], ["pull", named], ["pull", f"{data_dir.name}/gpu/other"]):
        monkeypatch.setattr("sys.argv", ["underhood-hub", *argv])
        hub.main()

    pushed, pulled, missing = capsys.readouterr().out.strip().splitlines()
    assert set(fake.repos[REPO]) == {"gpu/t4/curve.jsonl", "gpu/t4/nested/ckpt.pt"}
    assert f"{REPO}/gpu/t4" in pushed, pushed
    assert f"pulled into {run_folder}" in pulled, pulled
    assert "nothing on the Hub" in missing, missing


def test_the_cli_exits_with_the_reason_it_refused(
    hub_cli: tuple[FakeHub, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, data_dir = hub_cli
    monkeypatch.setattr("sys.argv", ["underhood-hub", "push", f"{data_dir.name}/gpu/missing"])

    with pytest.raises(SystemExit, match="no folder"):
        hub.main()
