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
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
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


def metadata(user: str, name: str) -> dict[str, object]:
    """kernel-metadata.json for a private script kernel on a T4 with internet."""
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
        "dataset_sources": [],
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


def push(run: Runner, user: str, name: str, command: str, keep: str, out_dir: Path) -> Path:
    """Write the kernel for this commit to out_dir/name and push it, which starts the run."""
    meta = metadata(user, name)
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


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run underhood on a free Kaggle T4.")
    parser.add_argument("action", choices=["push", "status", "pull"])
    parser.add_argument("name", help="the run's name: its kernel, and data/gpu/<name> by default")
    parser.add_argument("--user", default=None, help="your Kaggle username; config.KAGGLE_USER")
    parser.add_argument("--command", default=None, help="what to run in the checkout")
    parser.add_argument("--keep", default=None, help="the folder to hand back")
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
            folder = push(
                _run, user, name, command, args.keep or f"data/gpu/{name}", DATA_DIR / "kaggle"
            )
            print(f"pushed {folder}; `underhood-kaggle status {name}` to follow it")
        elif args.action == "status":
            print(status(_run, user, name))
        else:
            print(f"output in {pull(_run, user, name, DATA_DIR / 'kaggle' / name / 'output')}")
    except (ValueError, RuntimeError) as refused:
        sys.exit(str(refused))


if __name__ == "__main__":
    main()
