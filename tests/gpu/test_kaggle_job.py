"""Specification for gpu/kaggle_job.py, the script a Kaggle kernel runs, with its shell faked."""

from __future__ import annotations

import os
import sys
import types

import pytest

from underhood.gpu import kaggle_job

REPO_URL, SHA = "https://github.com/someone/underhood.git", "0123abc"
COMMAND, KEEP = "uv run underhood-train-gpu --name t4", "data/gpu/t4"


class Shell:
    """Records each command and answers with the exit code of the first rule it matches."""

    def __init__(self, failing: str | None = None, code: int = 3) -> None:
        self.ran: list[str] = []
        self.failing, self.code = failing, code

    def __call__(self, command: str) -> int:
        self.ran.append(command)
        return self.code if self.failing is not None and self.failing in command else 0

    def index(self, fragment: str) -> int:
        return next(i for i, command in enumerate(self.ran) if fragment in command)


def _main(shell: Shell, token: str | None) -> None:
    kaggle_job.main(REPO_URL, SHA, COMMAND, KEEP, run=shell, secret=lambda: token)


@pytest.mark.parametrize(
    ("token", "hub"),
    [
        pytest.param("hf_token", True, id="with an HF_TOKEN secret the run resumes from the Hub"),
        pytest.param(None, False, id="without one it starts fresh and keeps only Kaggle's copy"),
    ],
)
def test_a_job_checks_out_the_commit_runs_it_and_keeps_the_folder(
    token: str | None, hub: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    shell = Shell()

    _main(shell, token)

    fetch, checkout = shell.index("fetch"), shell.index("checkout")
    sync, command = shell.index("uv sync --locked"), shell.index(COMMAND)
    copy = shell.index(f"{kaggle_job.WORKING / KEEP}")
    assert REPO_URL in shell.ran[fetch] and SHA in shell.ran[fetch], shell.ran[fetch]
    assert "FETCH_HEAD" in shell.ran[checkout]
    assert fetch < checkout < sync < command < copy, shell.ran
    assert shell.ran[command].startswith(f"cd {kaggle_job.CHECKOUT} && "), "run in the checkout"
    hub_steps = [c for c in shell.ran if "underhood-hub" in c]
    if hub:
        pull, push = shell.index(f"underhood-hub pull {KEEP}"), shell.index(f"hub push {KEEP}")
        assert sync < pull < command < push, shell.ran
        assert os.environ.get("HF_TOKEN") == token, "huggingface_hub reads the token from here"
    else:
        assert hub_steps == [], hub_steps
        assert "HF_TOKEN" not in os.environ


@pytest.mark.parametrize(
    ("failing", "ran_command", "kept"),
    [
        pytest.param("uv sync", False, False, id="a failed setup stops before the command"),
        pytest.param(COMMAND, True, True, id="a failed command still hands back its folder"),
    ],
)
def test_a_job_exits_with_the_code_of_the_step_that_failed(
    failing: str, ran_command: bool, kept: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    shell = Shell(failing=failing, code=3)

    with pytest.raises(SystemExit) as exited:
        _main(shell, "hf_token")

    assert exited.value.code == 3
    assert any(COMMAND in c for c in shell.ran) is ran_command, shell.ran
    assert any(f"hub push {KEEP}" in c for c in shell.ran) is kept, shell.ran
    assert any(str(kaggle_job.WORKING / KEEP) in c for c in shell.ran) is kept, shell.ran


@pytest.mark.parametrize(
    ("module", "token"),
    [
        pytest.param(None, None, id="off Kaggle there is no secrets client"),
        pytest.param("hf_secret", "hf_secret", id="on Kaggle the secret is read"),
        pytest.param("", None, id="an empty secret counts as none"),
        pytest.param(KeyError("HF_TOKEN"), None, id="a secret never added counts as none"),
    ],
)
def test_hf_token(
    module: str | Exception | None, token: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if module is None:
        monkeypatch.setitem(sys.modules, "kaggle_secrets", None)  # import raises ImportError
    else:

        class UserSecretsClient:
            def get_secret(self, label: str) -> str:
                assert label == "HF_TOKEN"
                if isinstance(module, Exception):
                    raise module
                return module

        fake = types.ModuleType("kaggle_secrets")
        setattr(fake, "UserSecretsClient", UserSecretsClient)  # noqa: B010, assigning it is a pyright error
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake)

    assert kaggle_job.hf_token() == token
