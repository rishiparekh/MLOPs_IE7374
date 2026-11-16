"""LoRA written from scratch in PyTorch (no peft).

Instead of updating a weight W (out x in), keep W frozen and learn a low-rank
update:   W' = W + (alpha / r) * B @ A     A: (r x in), B: (out x r)
B starts at zero, so at step 0 the adapted model is exactly the base model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        # base can be a normal nn.Linear or the QuantLinear from quant.py
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scale = alpha / r
        self.enabled = True

        for p in self.base.parameters():
            p.requires_grad = False

        # put the adapters on the same device as the layer they wrap
        # (a QuantLinear has no .weight, so look at its codes instead)
        ref = base.weight if hasattr(base, "weight") else base.codes
        self.A = nn.Parameter(torch.empty(r, self.in_features, device=ref.device))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, device=ref.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same init nn.Linear uses
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            update = F.linear(F.linear(self.dropout(x), self.A), self.B)
            out = out + self.scale * update
        return out

    @torch.no_grad()
    def merged_weight(self):
        """W + (alpha/r) B A  as a plain tensor (only valid for an fp base)."""
        return self.base.weight + self.scale * (self.B @ self.A)


def apply_lora(model, target_names, r=8, alpha=16, dropout=0.0):
    """Wrap every module whose attribute name is in target_names."""
    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in target_names and hasattr(child, "in_features"):
                setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    return wrapped


def lora_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def freeze_all_but_lora(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.endswith(".A") or name.endswith(".B")


def set_lora_enabled(model, enabled):
    for layer in lora_layers(model):
        layer.enabled = enabled


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(".A") or k.endswith(".B")}


def load_lora_state_dict(model, state):
    missing = model.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys


@torch.no_grad()
def merge_lora(model):
    """Fold the adapters into the base weights so inference costs nothing extra.
    Only works when the base layers are ordinary fp nn.Linear (not QLoRA)."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                if not isinstance(child.base, nn.Linear):
                    raise ValueError("can't merge into a quantized base layer")
                child.base.weight.copy_(child.merged_weight())
                setattr(parent, name, child.base)
