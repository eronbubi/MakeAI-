"""Parameter-efficient training: LoRA, QLoRA (4/8-bit base via bitsandbytes), bottleneck adapters."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import BottleneckAdapter, MakeAIForCausalLM

LINEAR_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, in_f: int, out_f: int, r: int, alpha: float, dropout: float,
                 train_bias: bool = False):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        bias = getattr(base, "bias", None)
        if bias is not None:
            bias.requires_grad_(train_bias)

    def forward(self, x):
        out = self.base(x)
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        return out + F.linear(F.linear(self.drop(x), a), b) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B.float() @ self.lora_A.float()) * self.scaling


def _set(model: nn.Module, name: str, new: nn.Module):
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def quantize_linears(model: MakeAIForCausalLM, quant: str, compute_dtype=torch.bfloat16,
                     double_quant: bool = True, targets: Iterable[str] = LINEAR_TARGETS, device="cuda") -> int:
    """Replace transformer-block linears with bitsandbytes 8-bit or 4-bit (NF4/FP4) layers. Returns count."""
    import bitsandbytes as bnb
    if quant == "int8":
        odd = sorted({m.out_features for n_, m in model.named_modules()
                      if isinstance(m, nn.Linear) and n_.split(".")[-1] in targets and m.out_features % 32})
        if odd:
            raise ValueError(f"INT8 (LLM.int8) base quantization needs layer widths divisible by 32; this model has "
                             f"{odd}. bitsandbytes pads them in its tiled GPU layout and the backward pass fails. "
                             f"Use NF4/FP4 (QLoRA) or a 16-bit base instead.")
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or name.split(".")[-1] not in targets:
            continue
        w = mod.weight.data
        if quant in ("nf4", "fp4", "int4"):
            q = bnb.nn.Linear4bit(mod.in_features, mod.out_features, bias=mod.bias is not None,
                                  compute_dtype=compute_dtype, compress_statistics=double_quant,
                                  quant_type="fp4" if quant == "fp4" else "nf4", device="cpu")
            q.weight = bnb.nn.Params4bit(w.to(torch.float16 if compute_dtype == torch.float16 else torch.bfloat16).cpu(),
                                         requires_grad=False, compress_statistics=double_quant,
                                         quant_type="fp4" if quant == "fp4" else "nf4")
        elif quant == "int8":
            q = bnb.nn.Linear8bitLt(mod.in_features, mod.out_features, bias=mod.bias is not None,
                                    has_fp16_weights=False, threshold=6.0, device="cpu")
            q.weight = bnb.nn.Int8Params(w.to(torch.float16).cpu(), requires_grad=False, has_fp16_weights=False)
        else:
            raise ValueError(f"unknown quantization {quant}")
        if mod.bias is not None:
            q.bias = nn.Parameter(mod.bias.data.to(compute_dtype).cpu(), requires_grad=False)
        _set(model, name, q.to(device))
        n += 1
    return n


def apply_lora(model: MakeAIForCausalLM, r: int = 16, alpha: float = 32, dropout: float = 0.05,
               targets: Iterable[str] = LINEAR_TARGETS, bias: str = "none") -> int:
    """Freeze the model and wrap target linears with LoRA. ``bias``: none | all | lora_only."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = set(targets)
    n = 0
    for name, mod in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if leaf not in targets or isinstance(mod, LoRALinear):
            continue
        in_f = getattr(mod, "in_features", None)
        out_f = getattr(mod, "out_features", None)
        if in_f is None:
            continue
        dev = next((t.device for t in list(mod.parameters()) + list(mod.buffers())), torch.device("cpu"))
        lora = LoRALinear(mod, in_f, out_f, r, alpha, dropout, train_bias=bias in ("all", "lora_only"))
        lora.lora_A.data = lora.lora_A.data.to(dev)
        lora.lora_B.data = lora.lora_B.data.to(dev)
        _set(model, name, lora)
        n += 1
    if bias == "all":
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad_(True)
    return n


def apply_adapters(model: MakeAIForCausalLM, bottleneck: int = 64) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device
    h = model.cfg.hidden_size
    for layer in model.model.layers:
        layer.adapter_attn = BottleneckAdapter(h, bottleneck).to(dev)
        layer.adapter_mlp = BottleneckAdapter(h, bottleneck).to(dev)
    return 2 * len(model.model.layers)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the parameters that were trained (LoRA A/B, adapters, trainable biases)."""
    return {k: v.detach().cpu().contiguous() for k, v in model.named_parameters() if v.requires_grad}


def load_trainable(model: nn.Module, sd: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = [k for k in sd if k not in params]
    if missing:
        raise KeyError(f"adapter keys not in model: {missing[:5]}")
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))


def merge_lora_into_state_dict(base_sd: dict[str, torch.Tensor], adapter_sd: dict[str, torch.Tensor],
                               scaling: float) -> dict[str, torch.Tensor]:
    """Merge LoRA deltas into full-precision base weights (keys use MakeAI/HF names)."""
    out = dict(base_sd)
    for k in adapter_sd:
        if not k.endswith(".lora_A"):
            continue
        prefix = k[: -len(".lora_A")]
        a = adapter_sd[k].float()
        b = adapter_sd[prefix + ".lora_B"].float()
        wk = prefix + ".weight"
        w = out[wk]
        out[wk] = w.float() + (b @ a) * scaling   # fp32: the caller casts once to its compute dtype
    for k, v in adapter_sd.items():
        if k.endswith(".base.bias"):
            out[k.replace(".base.bias", ".bias")] = v
    return out


def adapter_keys_to_base(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalise wrapped-module names ('x.base.weight') back to plain names ('x.weight')."""
    return {k.replace(".base.", "."): v for k, v in sd.items()}
