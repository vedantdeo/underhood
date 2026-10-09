"""A run folder under data/ to and from a private Hugging Face model repo, so a run on Kaggle can
resume from its last checkpoint and hand its results back.

uv run underhood-hub push data/gpu/t4   # into <you>/underhood-runs, at gpu/t4
uv run underhood-hub pull data/gpu/t4   # back again; says so if the Hub has no such run

huggingface_hub is imported only when a call needs it; `hf auth login`, or HF_TOKEN, once.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from underhood import config
from underhood.gpu import DATA_DIR

if TYPE_CHECKING:
    from huggingface_hub import HfApi


class HubApi(Protocol):
    """The part of huggingface_hub.HfApi used here."""

    def whoami(self) -> Mapping[str, object]: ...

    def create_repo(self, repo_id: str, *, private: bool, exist_ok: bool) -> object: ...

    def upload_folder(self, *, repo_id: str, folder_path: Path, path_in_repo: str) -> object: ...

    def list_repo_files(self, repo_id: str) -> list[str]: ...

    def snapshot_download(
        self, repo_id: str, *, allow_patterns: str, local_dir: Path
    ) -> object: ...


def repo_id(api: HubApi) -> str:
    """The runs repo under the logged-in account."""
    return f"{api.whoami()['name']}/{config.HUB_RUNS_REPO}"


def _in_data(folder: Path, data_dir: Path) -> str:
    """folder's path inside data_dir, as the Hub names it; ValueError if it lies outside."""
    resolved, root = folder.resolve(), data_dir.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"{folder} is not a folder inside {data_dir}")
    return resolved.relative_to(root).as_posix()


def push(api: HubApi, folder: Path, data_dir: Path = DATA_DIR) -> str:
    """Upload folder to the runs repo at its path inside data_dir, creating the repo private on
    first use. Return where it went. ValueError for a missing or empty folder."""
    path = _in_data(folder, data_dir)
    if not folder.is_dir():
        raise ValueError(f"no folder at {folder}")
    if not any(folder.iterdir()):
        raise ValueError(f"{folder} is empty; there is nothing to keep")
    repo = repo_id(api)
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=path)
    return f"{repo}/{path}"


def pull(api: HubApi, folder: Path, data_dir: Path = DATA_DIR) -> bool:
    """Download the runs repo's copy of folder into it; False, writing nothing, when the Hub has
    no such run or no runs repo yet."""
    from huggingface_hub.errors import RepositoryNotFoundError

    path = _in_data(folder, data_dir)
    repo = repo_id(api)
    try:
        files = api.list_repo_files(repo)
    except RepositoryNotFoundError:
        return False
    if not any(name.startswith(f"{path}/") for name in files):
        return False
    api.snapshot_download(repo, allow_patterns=f"{path}/*", local_dir=data_dir)
    return True


def _api() -> HfApi:
    from huggingface_hub import HfApi

    return HfApi()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep a run folder on the Hugging Face Hub.")
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("folder", type=Path, help="a folder under data/, such as data/gpu/t4")
    args = parser.parse_args()
    folder: Path = args.folder.resolve()
    try:
        if args.action == "push":
            print(f"kept on the Hub at {push(_api(), folder, DATA_DIR)}")
        elif pull(_api(), folder, DATA_DIR):
            print(f"pulled into {folder}")
        else:
            print(f"nothing on the Hub for {folder}; starting fresh")
    except ValueError as refused:
        sys.exit(str(refused))


if __name__ == "__main__":
    main()
