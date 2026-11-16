"""Checks for the pieces written from scratch (LoRA, NF4 quantization, prompts).

These use tiny random layers/models, so they run in a few seconds and don't
need the SmolLM weights. Run from Lab_04/:  python -m pytest tests -v
"""
import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

import lora
import quant
from tickets import LABELS, few_shot_prompt, load_tickets, sample_per_class, zero_shot_prompt


def tiny_llama():
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


# ---------------------------------------------------------------- NF4 / quant

def test_nf4_table_has_16_levels():
    levels = quant.NF4_LEVELS
    assert len(levels) == 16
    assert levels.min() == -1.0 and levels.max() == 1.0
    assert 0.0 in levels
    assert torch.all(levels[1:] > levels[:-1])  # strictly increasing


@pytest.mark.parametrize("shape", [(64, 64), (30, 50)])  # 1500 isn't a multiple of 64
def test_roundtrip_shape_and_error(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape) * 0.05
    errs = {}
    for bits in (8, 4):
        q = quant.quantize(w, bits=bits, block_size=64)
        back = quant.dequantize(*q, bits=bits, block_size=64)
        assert back.shape == w.shape
        errs[bits] = (back - w).abs().mean().item()
    assert errs[8] < errs[4]  # more bits, less error
    assert errs[4] < 0.1 * w.abs().max().item()


def test_4bit_codes_are_packed_two_per_byte():
    w = torch.randn(64, 64)
    codes, absmax, _, _ = quant.quantize(w, bits=4, block_size=64)
    assert codes.dtype == torch.uint8
    assert codes.numel() == w.numel() // 2
    assert absmax.numel() == w.numel() // 64


def test_quantlinear_close_to_linear():
    torch.manual_seed(0)
    lin = nn.Linear(64, 48)
    qlin = quant.QuantLinear(lin, bits=8)
    x = torch.randn(5, 64)
    assert torch.allclose(lin(x), qlin(x), atol=0.05)
    assert len(list(qlin.parameters())) == 0  # nothing trainable in the base


def test_4bit_uses_much_less_memory_than_fp32():
    lin = nn.Linear(256, 256, bias=False)
    qlin = quant.QuantLinear(lin, bits=4, block_size=64)
    assert quant.tensor_bytes(qlin) < 0.2 * quant.tensor_bytes(lin)


# ----------------------------------------------------------------------- LoRA

def test_zero_init_B_gives_same_output_as_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    wrapped = lora.LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.equal(base(x), wrapped(x))


def test_disabled_adapter_gives_base_output():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    wrapped = lora.LoRALinear(base, r=4, alpha=8)
    nn.init.normal_(wrapped.B)  # make the adapter do something
    x = torch.randn(3, 16)
    assert not torch.allclose(base(x), wrapped(x))
    wrapped.enabled = False
    assert torch.equal(base(x), wrapped(x))


def test_only_lora_params_are_trainable():
    model = tiny_llama()
    n = lora.apply_lora(model, ("q_proj", "v_proj"), r=4)
    assert n == 4  # 2 layers x 2 targets
    lora.freeze_all_but_lora(model)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(name.endswith((".A", ".B")) for name in trainable)


def test_trainable_param_count_matches_formula():
    model = tiny_llama()
    r = 4
    lora.apply_lora(model, ("q_proj",), r=r)
    lora.freeze_all_but_lora(model)
    got = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert got == 2 * r * (32 + 32)  # 2 layers, A is r x 32 and B is 32 x r


def test_merge_matches_adapter_output():
    model = tiny_llama()
    lora.apply_lora(model, ("q_proj", "k_proj", "v_proj", "o_proj"), r=4, alpha=8)
    for layer in lora.lora_layers(model):
        nn.init.normal_(layer.B, std=0.1)  # pretend we trained it
    ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        before = model(ids).logits
        lora.merge_lora(model)
        after = model(ids).logits
    assert not lora.lora_layers(model)  # wrappers are gone
    assert torch.allclose(before, after, atol=1e-5)


def test_adapter_save_and_load_roundtrip():
    model = tiny_llama()
    lora.apply_lora(model, ("q_proj",), r=4)
    for layer in lora.lora_layers(model):
        nn.init.normal_(layer.B, std=0.1)
    state = lora.lora_state_dict(model)
    assert state and all(k.endswith((".A", ".B")) for k in state)

    fresh = tiny_llama()
    lora.apply_lora(fresh, ("q_proj",), r=4)
    lora.load_lora_state_dict(fresh, state)
    ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, fresh(ids).logits, atol=1e-6)


def test_lora_on_quantized_base_trains_and_cannot_merge():
    model = tiny_llama()
    quant.quantize_linears(model.model.layers, bits=4, block_size=32)
    lora.apply_lora(model, ("q_proj",), r=4)
    lora.freeze_all_but_lora(model)
    ids = torch.randint(0, 100, (2, 10))
    loss = model(ids, labels=ids).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    with pytest.raises(ValueError):
        lora.merge_lora(model)


# -------------------------------------------------------------- data / prompts

def test_split_is_balanced():
    train, test = load_tickets()
    assert len(train) == len(test) == 100
    assert train["label"].value_counts().eq(20).all()
    assert test["label"].value_counts().eq(20).all()
    assert not set(train["text"]) & set(test["text"])


def test_sample_per_class_is_balanced_and_repeatable():
    train, _ = load_tickets()
    a = sample_per_class(train, 3, seed=1)
    b = sample_per_class(train, 3, seed=1)
    assert a.equals(b)
    assert a["label"].value_counts().eq(3).all()


def test_prompts_end_with_category_cue():
    train, _ = load_tickets()
    shots = sample_per_class(train, 1, seed=0)
    zs = zero_shot_prompt("my wifi is down")
    fs = few_shot_prompt("my wifi is down", shots)
    assert zs.endswith("Category:") and fs.endswith("Category:")
    assert "my wifi is down" in zs and "my wifi is down" in fs
    assert fs.count("Category:") == len(shots) + 1  # one per example + the question
    assert all(label in zs for label in LABELS)
