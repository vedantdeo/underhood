"""Run an underhood command on a free Kaggle T4: push a kernel that runs this commit, ask after it,
and download what it kept.

uv run underhood-kaggle push t4     # TinyStories training into data/gpu/t4, or --command
uv run underhood-kaggle status t4
uv run underhood-kaggle pull t4     # into data/kaggle/t4/output

Shells out to the kaggle CLI (`uv run kaggle auth login` once). Add your HF token as the Kaggle
secret HF_TOKEN and the run resumes from, and hands back to, the Hub (gpu.hub).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from underhood import config
from underhood.gpu import DATA_DIR, kaggle_job

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
CODE_FILE = "run.py"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")


def slug(name: str) -> str:
    """The kernel's slug for a run name; ValueError for a name Kaggle would not keep as it is."""
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"run name {name!r}: lower-case letters, digits and dashes, at most 40, no leading dash"
        )
    return f"underhood-{name}"


def metadata(user: str, name: str, datasets: Sequence[str] = ()) -> dict[str, object]:
    """kernel-metadata.json for a private script kernel on a T4 with internet, with `datasets`
    (`user/slug` refs) attached under /kaggle/input."""
    return {
        "id": f"{user}/{slug(name)}",
        "title": slug(name),
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": config.KAGGLE_ACCELERATOR,
        "dataset_sources": list(datasets),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def launcher(sha: str, command: str, keep: str) -> str:
    """kaggle_job's source with this run's arguments, called when Kaggle runs it as a script."""
    params = {"repo_url": config.KAGGLE_REPO_URL, "sha": sha, "command": command, "keep": keep}
    return (
        f"{inspect.getsource(kaggle_job)}\n\nPARAMS = {params!r}\n\n"
        'if __name__ == "__main__":\n    main(**PARAMS)\n'
    )


def _call(run: Runner, argv: Sequence[str]) -> str:
    """argv's stdout; RuntimeError carrying its stderr when it fails."""
    done = run(argv)
    if done.returncode:
        raise RuntimeError(f"{' '.join(argv)} failed: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


def _pushed_commit(run: Runner) -> str:
    """HEAD, refusing uncommitted changes or a commit on no remote branch: Kaggle checks out the
    commit from GitHub, so it would run neither."""
    if _call(run, ["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise ValueError("uncommitted changes would not reach Kaggle; commit them first")
    sha = _call(run, ["git", "rev-parse", "HEAD"]).strip()
    if not _call(run, ["git", "branch", "-r", "--contains", sha]).strip():
        raise ValueError(f"commit {sha[:12]} is on no remote branch; push it first")
    return sha


def push(
    run: Runner,
    user: str,
    name: str,
    command: str,
    keep: str,
    out_dir: Path,
    datasets: Sequence[str] = (),
) -> Path:
    """Write the kernel for this commit to out_dir/name and push it, which starts the run."""
    meta = metadata(user, name, datasets)
    sha = _pushed_commit(run)
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / CODE_FILE).write_text(launcher(sha, command, keep))
    _call(run, ["kaggle", "kernels", "push", "-p", str(folder)])
    return folder


def status(run: Runner, user: str, name: str) -> str:
    """What Kaggle says about the run's kernel."""
    return _call(run, ["kaggle", "kernels", "status", f"{user}/{slug(name)}"]).strip()


def pull(run: Runner, user: str, name: str, dest: Path) -> Path:
    """Download the kernel's output, its log and the folder it kept, into dest."""
    _call(run, ["kaggle", "kernels", "output", f"{user}/{slug(name)}", "-p", str(dest)])
    return dest


def upload(
    run: Runner, user: str, name: str, folder: Path, wait: Callable[[float], None] = time.sleep
) -> str:
    """The files directly in `folder` as a private Kaggle dataset, created or given a new version,
    returned only once Kaggle has processed it, so a kernel pushed next can mount it; its ref.
    Subfolders, such as a run's output, are skipped. ValueError for a folder with none;
    RuntimeError if Kaggle fails to process it or it is not ready in time."""
    if not any(p.is_file() and p.name != "dataset-metadata.json" for p in folder.iterdir()):
        raise ValueError(f"no files to upload in {folder}")
    ref = f"{user}/{slug(name)}"
    meta = {"title": slug(name), "id": ref, "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    exists = run(["kaggle", "datasets", "status", ref]).returncode == 0
    verb = ["version", "-m", "underhood upload"] if exists else ["create"]
    started = datetime.now(UTC) - timedelta(seconds=config.KAGGLE_CLOCK_SKEW_SECONDS)
    _call(run, ["kaggle", "datasets", *verb, "-p", str(folder), "-r", "skip"])
    _wait_until_ready(run, ref, folder, started, wait)
    return ref


def _wait_until_ready(
    run: Runner, ref: str, folder: Path, started: datetime, wait: Callable[[float], None]
) -> None:
    """Poll the dataset's file listing until every file in `folder` is there, at its size, created
    since `started`."""
    wanted = {
        p.name: p.stat().st_size
        for p in folder.iterdir()
        if p.is_file() and p.name != "dataset-metadata.json"
    }
    waited = 0.0
    while True:
        listed = _listing(_call(run, ["kaggle", "datasets", "files", ref, "-v"]))
        if all(
            name in listed and listed[name][0] == size and listed[name][1] >= started
            for name, size in wanted.items()
        ):
            return
        if waited >= config.KAGGLE_DATASET_READY_SECONDS:
            raise RuntimeError(f"{ref} not ready after {waited:.0f}s: the new version is unlisted")
        wait(config.KAGGLE_DATASET_POLL_SECONDS)
        waited += config.KAGGLE_DATASET_POLL_SECONDS


def _listing(out: str) -> dict[str, tuple[int, datetime]]:
    """`kaggle datasets files -v` as name -> (size, creation time in UTC)."""
    rows = csv.DictReader(out.splitlines())
    return {
        row["name"]: (
            int(row["size"]),
            datetime.fromisoformat(row["creationDate"]).replace(tzinfo=UTC),
        )
        for row in rows
    }


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run underhood on a free Kaggle T4.")
    parser.add_argument("action", choices=["push", "status", "pull", "data"])
    parser.add_argument("name", help="the run's name: its kernel, and data/gpu/<name> by default")
    parser.add_argument("--user", default=None, help="your Kaggle username; config.KAGGLE_USER")
    parser.add_argument("--command", default=None, help="what to run in the checkout")
    parser.add_argument("--keep", default=None, help="the folder to hand back")
    parser.add_argument("--dataset", action="append", default=[], help="a dataset to attach")
    parser.add_argument("--folder", type=Path, help="data: the folder to upload")
    args = parser.parse_args()
    user: str | None = args.user or config.KAGGLE_USER
    try:
        if user is None:
            raise ValueError("no Kaggle user: set KAGGLE_USER in config, or pass --user")
        name: str = args.name
        if args.action == "push":
            command = args.command or (
                "uv run underhood-fetch-tinystories && "
                f"uv run underhood-train-gpu --gpu kaggle-t4 --name {name}"
            )
            keep = args.keep or f"data/gpu/{name}"
            attached = [f"{user}/{slug(d)}" for d in args.dataset]
            folder = push(_run, user, name, command, keep, DATA_DIR / "kaggle", attached)
            print(f"pushed {folder}; `underhood-kaggle status {name}` to follow it")
        elif args.action == "data":
            if args.folder is None:
                raise ValueError("data needs --folder")
            print(f"dataset {upload(_run, user, name, args.folder)}; attach with --dataset {name}")
        elif args.action == "status":
            print(status(_run, user, name))
        else:
            print(f"output in {pull(_run, user, name, DATA_DIR / 'kaggle' / name / 'output')}")
    except (ValueError, RuntimeError) as refused:
        sys.exit(str(refused))


if __name__ == "__main__":
    main()
