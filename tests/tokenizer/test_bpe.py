"""Specification for tokenizer/bpe.py. Make these green from the top down."""

from __future__ import annotations

import pytest

from underhood.tokenizer.bpe import BPETokenizer, get_stats, merge

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. The dog, being lazy, did not react. "
    "Quick foxes and lazy dogs: a tale as old as time, told again and again and again. "
) * 20


def test_get_stats_counts_adjacent_pairs() -> None:
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_get_stats_of_short_lists_is_empty() -> None:
    assert get_stats([]) == {}
    assert get_stats([7]) == {}


def test_merge_replaces_every_occurrence() -> None:
    assert merge([5, 6, 6, 7, 9, 1], (6, 7), 99) == [5, 6, 99, 9, 1]
    assert merge([1, 2, 1, 2, 1], (1, 2), 3) == [3, 3, 1]


def test_merge_leaves_input_untouched() -> None:
    ids = [1, 2, 1, 2]
    merge(ids, (1, 2), 3)
    assert ids == [1, 2, 1, 2]


def test_train_learns_exactly_the_requested_number_of_merges() -> None:
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert len(tok.merges) == 300 - 256
    assert len(tok.vocab) == 300


def test_first_merge_is_the_most_frequent_pair_and_vocab_concatenates_bytes() -> None:
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=257)
    (((a, b), new_id),) = tok.merges.items()
    assert new_id == 256
    assert tok.vocab[256] == tok.vocab[a] + tok.vocab[b]
    raw = list(SAMPLE.encode("utf-8"))
    assert get_stats(raw)[(a, b)] == max(get_stats(raw).values())


@pytest.mark.parametrize(
    "schedule",
    [
        pytest.param([300, 320], id="two calls resume where the first stopped"),
        pytest.param([280, 300, 320], id="three calls resume repeatedly"),
        pytest.param([320, 320], id="repeating a reached target is a no-op"),
        pytest.param([400, 500], id="a target the text cannot reach saturates identically"),
    ],
)
def test_training_in_steps_matches_training_in_one_call(schedule: list[int]) -> None:
    stepwise = BPETokenizer()
    for vocab_size in schedule:
        stepwise.train(SAMPLE, vocab_size)

    # lower targets are no-ops, so the one-shot equivalent is the schedule's high-water mark
    one_shot = BPETokenizer()
    one_shot.train(SAMPLE, max(schedule))

    assert stepwise.merges == one_shot.merges, (
        f"{len(stepwise.merges)} merges stepwise vs {len(one_shot.merges)} in one call"
    )
    assert stepwise.vocab == one_shot.vocab


def test_training_rejects_a_target_below_the_vocab_already_reached() -> None:
    tok = BPETokenizer()
    tok.train(SAMPLE, 320)
    before = dict(tok.merges), dict(tok.vocab)

    with pytest.raises(ValueError):
        tok.train(SAMPLE, 300)

    assert (tok.merges, tok.vocab) == before, "a rejected train() must not mutate the tokenizer"


def test_resumed_training_leaves_no_unreachable_vocab_ids() -> None:
    tok = BPETokenizer()
    tok.train(SAMPLE, 300)
    tok.train(SAMPLE, 320)

    reachable = set(tok.merges.values()) | set(range(256))
    orphans = sorted(set(tok.vocab) - reachable)
    assert not orphans, f"vocab ids no merge rule can produce: {orphans}"


def test_encode_decode_roundtrip_on_training_text() -> None:
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=320)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


@pytest.mark.parametrize(
    "text",
    ["", "a", "hello world", "naïve café — 東京 🚀", "unseen words like pterodactyl and quokka"],
)
def test_encode_decode_roundtrip_on_unseen_text(text: str) -> None:
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=320)
    assert tok.decode(tok.encode(text)) == text


def test_training_actually_compresses() -> None:
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=320)
    assert len(tok.encode(SAMPLE)) < 0.6 * len(SAMPLE.encode("utf-8"))


def test_untrained_tokenizer_is_plain_bytes() -> None:
    tok = BPETokenizer()
    assert tok.encode("hi") == [104, 105]
    assert tok.decode([104, 105]) == "hi"


def test_decode_survives_a_split_multibyte_character() -> None:
    tok = BPETokenizer()
    first_byte_of_euro = "€".encode()[0]
    assert tok.decode([first_byte_of_euro]) == "�"
