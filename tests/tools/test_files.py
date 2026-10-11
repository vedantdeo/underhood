"""Specification for tools/files.py: reads and creates in one sandbox, refusals as ValueError."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from underhood.tools.config import MAX_FILE_READ_CHARS, MAX_FILE_WRITE_CHARS
from underhood.tools.files import read_file, write_file


def test_read_file_reads_inside_the_sandbox(box: Path) -> None:
    assert read_file("notes.txt", sandbox=box) == "hello from the sandbox\n"


def _symlink_pointing_out(box: Path) -> str:
    os.symlink(box.parent / "outside.txt", box / "link.txt")
    return "link.txt"


@pytest.mark.parametrize(
    "escape",
    [
        pytest.param(lambda box: "../outside.txt", id="parent traversal"),
        pytest.param(
            lambda box: str((box.parent / "outside.txt").resolve()), id="an absolute path"
        ),
        pytest.param(_symlink_pointing_out, id="a symlink pointing out"),
    ],
)
def test_read_file_refuses_a_path_that_leaves_the_sandbox(
    box: Path, escape: Callable[[Path], str]
) -> None:
    with pytest.raises(ValueError, match="escapes the workspace"):
        read_file(escape(box), sandbox=box)


def test_read_file_turns_an_unreadable_file_into_a_short_error(box: Path) -> None:
    locked = box / "locked.txt"
    locked.write_text("private", encoding="utf-8")
    locked.chmod(0)
    try:
        with pytest.raises(ValueError, match="Could not read locked.txt"):
            read_file("locked.txt", sandbox=box)
    finally:
        locked.chmod(0o600)


def test_read_file_names_a_missing_file(box: Path) -> None:
    with pytest.raises(ValueError, match="^File not found: gone.txt$"):
        read_file("gone.txt", sandbox=box)


def test_read_file_directory_is_a_short_error_without_the_absolute_path(box: Path) -> None:
    (box / "sub").mkdir()
    with pytest.raises(ValueError, match="^sub is a directory, not a file$"):
        read_file("sub", sandbox=box)
    with pytest.raises(ValueError, match="^\\. is a directory, not a file$"):
        read_file("", sandbox=box)


@pytest.mark.parametrize(
    ("size", "truncated"),
    [
        pytest.param(MAX_FILE_READ_CHARS, False, id="exactly at the cap"),
        pytest.param(MAX_FILE_READ_CHARS + 500, True, id="over the cap"),
    ],
)
def test_read_file_truncates_only_past_the_cap(box: Path, size: int, truncated: bool) -> None:
    (box / "big.txt").write_text("x" * size, encoding="utf-8")
    content = read_file("big.txt", sandbox=box)

    marker = f"\n[truncated at {MAX_FILE_READ_CHARS} characters]"
    assert content.endswith(marker) is truncated
    assert len(content.removesuffix(marker)) == min(size, MAX_FILE_READ_CHARS)


@pytest.mark.parametrize(
    ("file_path", "content"),
    [
        pytest.param("new.txt", "fresh\n", id="a new file"),
        pytest.param("deep/er/new.txt", "nested", id="with parents that do not exist yet"),
        pytest.param("empty.txt", "", id="an empty file"),
        pytest.param("big.txt", "x" * MAX_FILE_WRITE_CHARS, id="exactly at the cap"),
    ],
)
def test_write_file_creates_inside_the_sandbox(box: Path, file_path: str, content: str) -> None:
    said = write_file(file_path, content, sandbox=box)

    assert (box / file_path).read_text(encoding="utf-8") == content
    assert said == f"Wrote {len(content)} characters to {file_path}"


def _dangling_symlink_pointing_out(box: Path) -> str:
    os.symlink(box.parent / "planted.txt", box / "link.txt")
    return "link.txt"


@pytest.mark.parametrize(
    ("target", "says"),
    [
        pytest.param(lambda box: "../planted.txt", "escapes the workspace", id="parent traversal"),
        pytest.param(
            lambda box: str((box.parent / "planted.txt").resolve()),
            "escapes the workspace",
            id="an absolute path",
        ),
        pytest.param(
            _dangling_symlink_pointing_out, "escapes the workspace", id="a symlink pointing out"
        ),
        pytest.param(lambda box: "notes.txt", "notes.txt already exists", id="an existing file"),
        pytest.param(lambda box: "", "^\\. is a directory, not a file$", id="the sandbox itself"),
        pytest.param(
            lambda box: "notes.txt/inner.txt", "Could not write", id="a parent that is a file"
        ),
    ],
)
def test_write_file_refuses_without_touching_anything(
    box: Path, target: Callable[[Path], str], says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        write_file(target(box), "planted", sandbox=box)

    assert not (box.parent / "planted.txt").exists()
    assert (box / "notes.txt").read_text(encoding="utf-8") == "hello from the sandbox\n"


def test_write_file_refuses_content_over_the_cap(box: Path) -> None:
    with pytest.raises(ValueError, match=f"the cap is {MAX_FILE_WRITE_CHARS}"):
        write_file("big.txt", "x" * (MAX_FILE_WRITE_CHARS + 1), sandbox=box)
    assert not (box / "big.txt").exists()


def test_write_file_turns_an_unwritable_directory_into_a_short_error(box: Path) -> None:
    locked = box / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(ValueError, match="^Could not write locked/new.txt: Permission denied$"):
            write_file("locked/new.txt", "x", sandbox=box)
    finally:
        locked.chmod(0o700)
