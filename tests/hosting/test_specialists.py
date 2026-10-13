"""hosting/specialists.json, what hostllm hosts for aigent: its shape, and one canonical order."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

SPECIALISTS = Path(__file__).resolve().parents[2] / "hosting" / "specialists.json"


def load() -> dict[str, object]:
    return cast(dict[str, object], json.loads(SPECIALISTS.read_text(encoding="utf-8")))


def test_it_names_a_base_a_rank_and_each_adapter_by_hub_repo() -> None:
    spec = load()
    adapters = cast(dict[str, str], spec["adapters"])

    assert set(spec) == {"adapters", "base_model", "max_lora_rank"}
    assert isinstance(spec["base_model"], str) and isinstance(spec["max_lora_rank"], int)
    for name, repo in adapters.items():
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) and name != "base", name
        assert re.fullmatch(r"[\w.-]+/[\w.-]+", repo), repo


def test_it_is_written_in_canonical_order() -> None:
    text = SPECIALISTS.read_text(encoding="utf-8")

    assert text == json.dumps(load(), indent=2, sort_keys=True) + "\n", (
        "keys sorted, two-space indent"
    )
