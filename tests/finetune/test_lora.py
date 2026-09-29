"""Specification for finetune/lora.py: the layer, folding it back in, then fitting it to a model."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.conftest import GptFactory, TinyDims
from underhood.finetune.lora import LoRALinear, apply_lora
from underhood.model.gpt import GPT

IN, OUT, RANK, ALPHA = 12, 10, 4, 8.0
PROJECTIONS = ("q_proj", "v_proj")


def _layer(
    bias: bool = True, dtype: torch.dtype = torch.float32, dropout: float = 0.0
) -> LoRALinear:
    """Dropout off unless asked for, so the layer's arithmetic is exact."""
    torch.manual_seed(0)
    base = nn.Linear(IN, OUT, bias=bias, dtype=dtype)
    return LoRALinear(base, rank=RANK, alpha=ALPHA, dropout=dropout)


def _trained(layer: LoRALinear) -> LoRALinear:
    """Give B values, as training would, so the update is no longer zero."""
    with torch.no_grad():
        layer.lora_b.normal_()
    return layer


def test_the_update_has_a_low_rank_shape_and_a_scale() -> None:
    layer = _layer()
    assert layer.lora_a.shape == (RANK, IN)
    assert layer.lora_b.shape == (OUT, RANK)
    assert layer.scale == ALPHA / RANK


def test_at_init_the_layer_computes_exactly_what_its_base_did() -> None:
    """B starts at zero, so fine-tuning begins from the pretrained function, not a perturbed one."""
    layer = _layer()
    x = torch.randn(3, 5, IN)
    assert torch.equal(layer(x), layer.base(x))


def test_only_the_update_trains() -> None:
    layer = _layer()
    trainable = {name for name, p in layer.named_parameters() if p.requires_grad}
    assert trainable == {"lora_a", "lora_b"}
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == RANK * (IN + OUT)


def test_the_first_step_moves_b_and_not_a() -> None:
    """A reaches the output only through B, so while B is zero A's gradient is zero too."""
    layer = _layer()
    layer(torch.randn(4, IN)).square().sum().backward()
    assert layer.lora_b.grad is not None and torch.count_nonzero(layer.lora_b.grad) > 0
    assert layer.lora_a.grad is not None and torch.count_nonzero(layer.lora_a.grad) == 0
    assert layer.base.weight.grad is None, "the base is frozen"


def test_the_update_is_scale_times_x_through_a_then_b() -> None:
    layer = _trained(_layer())
    x = torch.randn(6, IN)
    expected = layer.base(x) + (ALPHA / RANK) * (x @ layer.lora_a.T @ layer.lora_b.T)
    assert torch.allclose(layer(x), expected, atol=1e-5)


@pytest.mark.parametrize(
    "bias", [pytest.param(True, id="with a bias"), pytest.param(False, id="without one")]
)
def test_merging_folds_the_update_into_one_plain_linear(bias: bool) -> None:
    layer = _trained(_layer(bias=bias))
    before = layer.base.weight.detach().clone()
    x = torch.randn(4, IN)

    merged = layer.merge()

    assert type(merged) is nn.Linear
    assert torch.allclose(merged(x), layer(x), atol=1e-5)
    assert (merged.bias is None) is not bias
    assert torch.equal(layer.base.weight, before), "merging returns a new layer; this one is kept"


def test_the_merged_update_has_exactly_the_adapters_rank() -> None:
    """What makes it cheap: the update to a 10x12 weight is a rank-4 matrix, 88 numbers not 120."""
    layer = _trained(_layer())
    update = layer.merge().weight - layer.base.weight
    assert int(torch.linalg.matrix_rank(update.detach())) == RANK


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [
        pytest.param(torch.float32, 1e-5, id="a float32 base"),
        pytest.param(torch.bfloat16, 5e-2, id="a bfloat16 base, which bf16 adapters would stall"),
    ],
)
def test_the_update_trains_in_float32_and_answers_in_the_bases_dtype(
    dtype: torch.dtype, atol: float
) -> None:
    layer = _trained(_layer(dtype=dtype))
    x = torch.randn(4, IN, dtype=dtype)

    assert layer.lora_a.dtype == layer.lora_b.dtype == torch.float32
    assert layer(x).dtype == dtype
    merged = layer.merge()
    assert merged.weight.dtype == dtype
    assert torch.allclose(merged(x).float(), layer(x).float(), atol=atol)


@pytest.mark.parametrize(
    ("training", "updated"),
    [
        pytest.param(True, False, id="training drops the whole update at p=1"),
        pytest.param(False, True, id="evaluating never drops"),
    ],
)
def test_dropout_touches_only_the_updates_input_and_only_in_training(
    training: bool, updated: bool
) -> None:
    layer = _trained(_layer(dropout=1.0)).train(training)
    x = torch.randn(4, IN)
    assert torch.equal(layer(x), layer.base(x)) is not updated


def test_apply_lora_wraps_each_target_in_every_block(tiny_gpt: GPT, dims: TinyDims) -> None:
    wrapped = apply_lora(tiny_gpt, targets=PROJECTIONS, rank=2, alpha=4.0)

    expected = [f"blocks.{i}.attn.{name}" for i in range(dims.n_layers) for name in PROJECTIONS]
    assert wrapped == expected
    for path in expected:
        assert isinstance(tiny_gpt.get_submodule(path), LoRALinear), path
    assert type(tiny_gpt.get_submodule("blocks.0.attn.k_proj")) is nn.Linear, "k is not a target"


def test_apply_lora_freezes_everything_but_the_updates(tiny_gpt: GPT, dims: TinyDims) -> None:
    wrapped = apply_lora(tiny_gpt, targets=PROJECTIONS, rank=2, alpha=4.0)

    trainable = {name for name, p in tiny_gpt.named_parameters() if p.requires_grad}
    assert trainable == {f"{path}.lora_{ab}" for path in wrapped for ab in "ab"}
    counted = sum(p.numel() for p in tiny_gpt.parameters() if p.requires_grad)
    assert counted == len(wrapped) * 2 * (dims.d_model + dims.d_model)


def test_a_model_with_fresh_updates_answers_exactly_as_before(
    make_gpt: GptFactory, dims: TinyDims
) -> None:
    base, adapted = make_gpt(), make_gpt()
    apply_lora(adapted, rank=2, alpha=4.0)
    idx = torch.randint(0, dims.vocab_size, (2, dims.block_size))
    assert torch.equal(adapted(idx)[0], base(idx)[0])


def test_a_target_that_matches_nothing_is_a_mistake_not_a_no_op(tiny_gpt: GPT) -> None:
    with pytest.raises(ValueError, match="query"):
        apply_lora(tiny_gpt, targets=("q_proj", "query"))


def test_fine_tuning_lowers_the_loss_and_leaves_every_pretrained_weight_alone(
    make_gpt: GptFactory, dims: TinyDims
) -> None:
    """The whole point, end to end: the adapters learn and the model underneath does not move."""
    model = make_gpt()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    apply_lora(model, targets=PROJECTIONS, rank=4, alpha=8.0)
    torch.manual_seed(1)
    idx = torch.randint(0, dims.vocab_size, (4, dims.block_size))
    targets = torch.roll(idx, -1, dims=1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)

    losses: list[float] = []
    for _ in range(30):
        _, loss = model(idx, targets)
        assert loss is not None
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0] - 0.1, f"loss went {losses[0]:.3f} -> {losses[-1]:.3f}"
    after = {
        name.replace(".base.", "."): p
        for name, p in model.named_parameters()
        if ".lora_" not in name
    }
    assert after.keys() == before.keys()
    for name, weight in before.items():
        assert torch.equal(after[name], weight), f"{name} moved"
