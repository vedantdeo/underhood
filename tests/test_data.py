"""Specification for data.py's TinyStories encoder: stories split on the separator, each encoded
and ended with <|endoftext|>, written as uint16 and only renamed into place once complete.
tiktoken is replaced by a fake, so nothing is downloaded."""

from __future__ import annotations

from pathlib import Path

import pytest

from underhood import data

EOT = 50256


class _Encoding:
    """GPT-2's shape without its vocabulary: a story encodes as its length, then a high id."""

    eot_token = EOT

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_ordinary_batch(self, texts: list[str]) -> list[list[int]]:
        self.calls.append(texts)
        return [[len(text), 40000] for text in texts]


@pytest.fixture
def encoding(monkeypatch: pytest.MonkeyPatch) -> _Encoding:
    fake = _Encoding()
    monkeypatch.setattr(data.tiktoken, "get_encoding", lambda name: fake)
    monkeypatch.setattr(data, "ENCODE_BATCH", 2)
    return fake


def _write(path: Path, stories: list[str]) -> Path:
    path.write_text(f"\n{data.STORY_SEPARATOR}\n".join(stories) + "\n", encoding="utf-8")
    return path


def test_every_story_is_encoded_and_ended_with_endoftext(
    tmp_path: Path, encoding: _Encoding
) -> None:
    source = _write(tmp_path / "train.txt", ["Once.", "Then  more.", "The end."])
    target = tmp_path / "train.bin"

    data._encode(source, target)

    ids = data.read_ids(target).tolist()
    assert ids == [5, 40000, EOT, 11, 40000, EOT, 8, 40000, EOT], "uint16 holds ids past 32,767"
    assert encoding.calls == [["Once.", "Then  more."], ["The end."]], "batched, stripped"
    assert not target.with_suffix(".partial").exists(), "renamed into place once complete"


def test_an_empty_story_is_skipped_rather_than_encoded_as_a_bare_endoftext(
    tmp_path: Path, encoding: _Encoding
) -> None:
    source = _write(tmp_path / "train.txt", ["Once.", "   ", "Twice."])

    data._encode(source, tmp_path / "train.bin")

    assert data.read_ids(tmp_path / "train.bin").tolist().count(EOT) == 2
    assert all("" not in texts for texts in encoding.calls)


@pytest.mark.parametrize(
    ("text", "stories"),
    [
        pytest.param("one\n<|endoftext|>\ntwo\n", ["one", "two"], id="two stories"),
        pytest.param("one\n<|endoftext|>\n", ["one"], id="a separator at the very end"),
        pytest.param("one\ntwo lines\n", ["one\ntwo lines"], id="no separator at all"),
    ],
)
def test_stories_split_on_the_separator_line(tmp_path: Path, text: str, stories: list[str]) -> None:
    path = tmp_path / "s.txt"
    path.write_text(text, encoding="utf-8")

    assert [story for batch in data._stories(path) for story in batch] == stories
