"""Specification for gpu/kaggle_job.py, the script a Kaggle kernel runs, with its shell faked."""

from __future__ import annotations

import os
import sys
import types

import pytest

from underhood.gpu import kaggle_job

REPO_URL, SHA = "https://github.com/someone/underhood.git", "0123abc"
COMMAND, KEEP = "uv run underhood-train-gpu --name t4", "data/gpu/t4"
ENV = ("HF_TOKEN", "PYTHONUNBUFFERED", "UV_NO_SYNC")


class Shell:
    """Records each command and the job's variables as it ran; fails the one matching `failing`."""

    def __init__(self, failing: str | None = None, code: int = 3) -> None:
        self.ran: list[str] = []
        self.env: list[dict[str, str | None]] = []
        self.failing, self.code = failing, code

    def __call__(self, command: str) -> int:
        self.ran.append(command)
        self.env.append({name: os.environ.get(name) for name in ENV})
        return self.code if self.failing is not None and self.failing in command else 0

    def index(self, fragment: str) -> int:
        return next(i for i, command in enumerate(self.ran) if fragment in command)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job writes to os.environ; start each test without its variables and restore after."""
    for name in ENV:
        monkeypatch.setenv(name, "")  # delenv alone records nothing for an unset variable
        monkeypatch.delenv(name)


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
    "name",
    [
        pytest.param("PYTHONUNBUFFERED", id="prints reach Kaggle's log as they happen"),
        pytest.param("UV_NO_SYNC", id="uv run keeps the train group setup installed"),
    ],
)
def test_every_step_runs_with(name: str) -> None:
    shell = Shell()

    _main(shell, None)

    seen = [env[name] for env in shell.env]
    assert seen == ["1"] * len(shell.ran), list(zip(shell.ran, seen, strict=True))


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
    shell = Shell(failing=failing, code=3)

    with pytest.raises(SystemExit) as exited:
        _main(shell, "hf_token")

    assert exited.value.code == 3
    assert any(COMMAND in c for c in shell.ran) is ran_command, shell.ran
    assert any(f"hub push {KEEP}" in c for c in shell.ran) is kept, shell.ran
    assert any(str(kaggle_job.WORKING / KEEP) in c for c in shell.ran) is kept, shell.ran


@pytest.mark.parametrize(
    ("module", "token", "said"),
    [
        pytest.param(None, None, "ModuleNotFoundError", id="off Kaggle there is no secrets client"),
        pytest.param("hf_secret", "hf_secret", "", id="on Kaggle the secret is read, quietly"),
        pytest.param("", None, "empty", id="an empty secret counts as none"),
        pytest.param(
            KeyError("not attached"), None, "KeyError: 'not attached'", id="a secret not attached"
        ),
    ],
)
def test_hf_token(
    module: str | Exception | None,
    token: str | None,
    said: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    out = capsys.readouterr().out
    assert (said in out) if said else out == "", f"the log says {out!r}, not why there is no token"
