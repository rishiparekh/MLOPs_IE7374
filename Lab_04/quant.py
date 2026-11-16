"""Block-wise weight quantization for the QLoRA part of the lab.

The base weights are stored in 4 bit (NF4) or 8 bit and dequantized back to
fp32 inside forward(). The LoRA adapters on top stay in fp32 and are the only
thing that gets trained.

Blocks: the flattened weight is cut into blocks of `block_size` values, and
each block is scaled by its own absmax so the values land in [-1, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# The 16 NF4 levels from the QLoRA paper (Dettmers et al. 2023). They are
# quantiles of a standard normal, so they are denser near 0 where most
# weights live.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141029358, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def quantize(weight, bits=4, block_size=64):
    """Returns (codes, absmax, shape, n_pad). codes is uint8 for 4 bit (two
    values packed per byte) and int8 for 8 bit."""
    shape = weight.shape
    flat = weight.detach().float().flatten()
    n_pad = (-flat.numel()) % block_size
    if n_pad:
        flat = torch.cat([flat, flat.new_zeros(n_pad)])
    blocks = flat.view(-1, block_size)
    levels = NF4_LEVELS.to(flat.device)

    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normed = blocks / absmax  # now in [-1, 1]

    if bits == 4:
        # nearest of the 16 levels, found block by block to keep memory small
        idx = torch.empty(normed.shape, dtype=torch.uint8, device=flat.device)
        for start in range(0, normed.shape[0], 4096):
            chunk = normed[start:start + 4096]
            idx[start:start + 4096] = (chunk.unsqueeze(-1) - levels).abs().argmin(-1).to(torch.uint8)
        idx = idx.flatten()
        codes = (idx[0::2] << 4) | idx[1::2]  # pack two 4-bit codes into a byte
    elif bits == 8:
        codes = torch.round(normed * 127).to(torch.int8).flatten()
    else:
        raise ValueError("bits must be 4 or 8")
    return codes, absmax.squeeze(1), shape, n_pad


def dequantize(codes, absmax, shape, n_pad, bits=4, block_size=64):
    if bits == 4:
        hi = codes >> 4
        lo = codes & 0x0F
        idx = torch.stack([hi, lo], dim=1).flatten().long()
        normed = NF4_LEVELS.to(codes.device)[idx]
    else:
        normed = codes.float() / 127
    blocks = normed.view(-1, block_size) * absmax.unsqueeze(1)
    flat = blocks.flatten()
    if n_pad:
        flat = flat[:-n_pad]
    return flat.view(shape)


class QuantLinear(nn.Module):
    """Drop-in replacement for a frozen nn.Linear with quantized weights."""

    def __init__(self, linear, bits=4, block_size=64):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.block_size = block_size
        codes, absmax, shape, n_pad = quantize(linear.weight, bits, block_size)
        self.shape, self.n_pad = shape, n_pad
        # buffers, not parameters: they are never trained
        self.register_buffer("codes", codes)
        self.register_buffer("absmax", absmax)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

    def weight_fp32(self):
        return dequantize(self.codes, self.absmax, self.shape, self.n_pad,
                          self.bits, self.block_size)

    def forward(self, x):
        # dequantize on the fly every call -- this is the QLoRA trade:
        # less memory for the weights, a bit more compute per step
        return F.linear(x, self.weight_fp32().to(x.dtype), self.bias)


def quantize_linears(module, bits=4, block_size=64, skip=("lm_head",)):
    """Replace every nn.Linear under `module` with a QuantLinear."""
    count = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(parent, name, QuantLinear(child, bits, block_size))
                count += 1
    return count


def tensor_bytes(module):
    """Bytes used by the parameters and buffers of a module."""
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total
