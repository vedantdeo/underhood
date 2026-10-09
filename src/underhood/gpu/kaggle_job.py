"""The script a Kaggle kernel runs: check out underhood at one commit, run a command, and hand back
the folder it writes, to /kaggle/working and, given an HF_TOKEN secret, to the Hub.

gpu.kaggle ships this file's source with its arguments appended. It imports nothing from underhood,
which is not installed until it runs, so every setting arrives as an argument and it has no config.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

CHECKOUT = Path("/tmp/underhood")  # outside /kaggle/working, which Kaggle saves whole as output
WORKING = Path("/kaggle/working")


def hf_token() -> str | None:
    """The HF_TOKEN secret from Kaggle's secrets client; None off Kaggle or if never added."""
    try:
        secret = import_module("kaggle_secrets").UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # the client raises its own errors for a secret not attached to the kernel
        return None
    return secret if isinstance(secret, str) and secret else None


def _shell(command: str) -> int:
    print(f"$ {command}", flush=True)
    return subprocess.run(command, shell=True).returncode


def main(
    repo_url: str,
    sha: str,
    command: str,
    keep: str,
    run: Callable[[str], int] = _shell,
    secret: Callable[[], str | None] = hf_token,
) -> None:
    """Set up, pull keep from the Hub, run command, then push keep and copy it to WORKING even if
    the command failed. Exit with the code of the step that failed."""
    token = secret()
    if token is not None:
        os.environ["HF_TOKEN"] = token
    inside = f"cd {CHECKOUT} && "
    setup = [
        f"git init -q {CHECKOUT}",
        f"git -C {CHECKOUT} fetch -q --depth 1 {repo_url} {sha}",
        f"git -C {CHECKOUT} checkout -q FETCH_HEAD",
        "pip install -q uv",
        f"{inside}uv sync --locked --no-default-groups --group train",
    ]
    hand_back = [f"mkdir -p {WORKING / keep} && cp -r {CHECKOUT / keep}/. {WORKING / keep}/"]
    if token is not None:
        setup.append(f"{inside}uv run underhood-hub pull {keep}")
        hand_back.insert(0, f"{inside}uv run underhood-hub push {keep}")
    for step in setup:
        if code := run(step):
            raise SystemExit(code)
    code = run(f"{inside}{command}")
    for step in hand_back:
        run(step)
    if code:
        raise SystemExit(code)
