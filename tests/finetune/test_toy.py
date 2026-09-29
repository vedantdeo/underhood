"""The toy fine-tune's settings reach mlx_lm as meant. Scaffolding, so green from the start."""

from __future__ import annotations

from pathlib import Path

import pytest
from mlx_lm import lora as mlx_lora

from underhood import config
from underhood.finetune.toy import arguments, main


def test_every_setting_mlx_lm_reads_is_present() -> None:
    assert set(mlx_lora.CONFIG_DEFAULTS) <= set(vars(arguments()))


def test_the_adapter_means_what_the_from_scratch_layer_means() -> None:
    """mlx scales by `scale` directly; the layer in lora.py by alpha / rank."""
    lora = arguments().lora_parameters
    assert lora["rank"] == config.LORA_RANK
    assert lora["scale"] == config.LORA_ALPHA / config.LORA_RANK
    assert lora["dropout"] == config.LORA_DROPOUT


def test_the_adapter_sits_on_the_projections_apply_lora_wraps_in_every_layer() -> None:
    """mlx's default adapts every linear in the last 16 blocks; apply_lora, the targets in all."""
    settings = arguments()
    assert settings.num_layers == -1, "mlx reads -1 as every layer"
    keys = settings.lora_parameters["keys"]
    assert [key.rsplit(".", 1)[-1] for key in keys] == list(config.LORA_TARGETS)


def test_only_the_answer_is_trained_on() -> None:
    """The prompt is the same schema instruction on every row; learning it teaches nothing."""
    settings = arguments()
    assert settings.mask_prompt is True
    assert settings.train is True


def test_training_without_rows_says_where_they_come_from(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="aigent.extraction.synthetic"):
        main(["--data", str(tmp_path)])
