"""Reading and creating files inside one sandbox directory, which a model's paths cannot leave."""

from __future__ import annotations

from pathlib import Path

from underhood.tools.config import MAX_FILE_READ_CHARS, MAX_FILE_WRITE_CHARS


def _in_sandbox(file_path: str, sandbox: Path) -> Path:
    """`file_path` resolved under `sandbox`, refusing one that escapes it or names a directory."""
    resolved = (sandbox / file_path).resolve()
    if not resolved.is_relative_to(sandbox.resolve()):
        raise ValueError(f"path escapes the workspace: {file_path}")
    if resolved.is_dir():
        raise ValueError(f"{file_path or '.'} is a directory, not a file")
    return resolved


def read_file(file_path: str, sandbox: Path) -> str:
    """Read the contents of a file and return it as a string."""
    try:
        resolved = _in_sandbox(file_path, sandbox)
        with open(resolved, encoding="utf-8", errors="replace") as f:
            r = f.read(MAX_FILE_READ_CHARS + 1)  # one extra character tells us whether it was cut
        if len(r) > MAX_FILE_READ_CHARS:
            r = r[:MAX_FILE_READ_CHARS] + f"\n[truncated at {MAX_FILE_READ_CHARS} characters]"
        return r
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {file_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {file_path}: {exc}") from exc


def write_file(file_path: str, content: str, sandbox: Path) -> str:
    """Create a new file, and any missing parent directories, inside the sandbox. Never
    overwrites: a sandbox may hold fixtures someone tracks."""
    if len(content) > MAX_FILE_WRITE_CHARS:
        raise ValueError(f"content is {len(content)} characters; the cap is {MAX_FILE_WRITE_CHARS}")
    try:
        resolved = _in_sandbox(file_path, sandbox)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" refuses an existing file atomically, where a check-then-write would race
            with open(resolved, "x", encoding="utf-8") as f:
                f.write(content)
        except FileExistsError as exc:
            raise ValueError(f"{file_path} already exists; pick a new name") from exc
    except OSError as exc:
        raise ValueError(f"Could not write {file_path}: {exc.strerror}") from exc
    return f"Wrote {len(content)} characters to {file_path}"
