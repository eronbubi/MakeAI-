"""MakeAI decoder-only transformer.

Module and parameter names deliberately follow the Hugging Face Llama/Qwen2
layout (``model.layers.N.self_attn.q_proj`` ...). That way weights trained here
export 1:1 to HF safetensors/GGUF, and Llama-, Mistral- and Qwen2-family
checkpoints import straight into this implementation without a conversion step.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import GATED, ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def make_norm(cfg: ModelConfig) -> nn.Module:
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
    return RMSNorm(cfg.hidden_size, cfg.norm_eps)


class Rotary(nn.Module):
    """Rotary position embedding with optional linear or NTK-aware scaling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.head_dim
        theta = cfg.rope_theta
        self.linear_factor = 1.0
        if cfg.rope_scaling == "ntk" and cfg.rope_scaling_factor > 1:
            theta = theta * cfg.rope_scaling_factor ** (d / (d - 2))
        elif cfg.rope_scaling == "linear" and cfg.rope_scaling_factor > 1:
            self.linear_factor = cfg.rope_scaling_factor
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._len = 0
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None

    def get(self, length: int, device, dtype):
        if self._cos is None or length > self._len or self._cos.device != device:
            n = max(length, 256)
            t = torch.arange(n, device=device, dtype=torch.float32) / self.linear_factor
            freqs = torch.outer(t, self.inv_freq.to(device).float())
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos, self._sin, self._len = emb.cos(), emb.sin(), n
        return self._cos[:length].to(dtype), self._sin[:length].to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    cos = cos[None, None]
    sin = sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def alibi_slopes(n_heads: int) -> torch.Tensor:
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if math.log2(n_heads).is_integer():
        s = pow2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        s = pow2(closest) + pow2(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(s, dtype=torch.float32)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h, hd = cfg.hidden_size, cfg.head_dim
        self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.q_proj = nn.Linear(h, cfg.n_heads * hd, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(h, cfg.n_kv_heads * hd, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(h, cfg.n_kv_heads * hd, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(cfg.n_heads * hd, h, bias=cfg.bias)
        if cfg.pos_encoding == "alibi":
            self.register_buffer("slopes", alibi_slopes(cfg.n_heads), persistent=False)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        B, T, _ = x.shape
        hd = self.cfg.head_dim
        q = self.q_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, hd).transpose(1, 2)
        past_len = past[0].shape[2] if past is not None else 0
        if cos is not None:
            q, k = apply_rotary(q, k, cos[past_len:past_len + T], sin[past_len:past_len + T])
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        cache = (k, v) if use_cache else None
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        S = k.shape[2]
        alibi = self.cfg.pos_encoding == "alibi"
        window = self.cfg.sliding_window if self.cfg.attention_type == "sliding_window" else 0
        mask, causal = None, False
        if not alibi and not window and past_len == 0:
            causal = True          # also correct for T == 1; a plain bool keeps ONNX tracing dynamic
        elif alibi or window or T > 1:
            qpos = torch.arange(past_len, past_len + T, device=x.device)[:, None]
            kpos = torch.arange(S, device=x.device)[None, :]
            allowed = kpos <= qpos
            if window:
                allowed &= kpos > qpos - window
            mask = torch.zeros(T, S, device=x.device, dtype=q.dtype).masked_fill(~allowed, float("-inf"))
            if alibi:
                mask = mask[None] + (self.slopes.to(q.dtype)[:, None, None] * (kpos - qpos).to(q.dtype)[None])
        p = self.cfg.attention_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p, is_causal=causal)
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * hd)
        return self.o_proj(y), cache


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.kind = cfg.activation
        h, i = cfg.hidden_size, cfg.intermediate_size
        if self.kind in GATED:
            self.gate_proj = nn.Linear(h, i, bias=cfg.bias)
        self.up_proj = nn.Linear(h, i, bias=cfg.bias)
        self.down_proj = nn.Linear(i, h, bias=cfg.bias)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        if self.kind == "geglu":
            return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[self.kind]
        return self.down_proj(act(self.up_proj(x)))


class BottleneckAdapter(nn.Module):
    """Houlsby-style adapter: x + up(act(down(x))), zero-initialised up-projection."""

    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = make_norm(cfg)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = make_norm(cfg)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.adapter_attn: Optional[BottleneckAdapter] = None
        self.adapter_mlp: Optional[BottleneckAdapter] = None

    def forward(self, x, cos, sin, past=None, use_cache=False):
        a, cache = self.self_attn(self.input_layernorm(x), cos, sin, past, use_cache)
        if self.adapter_attn is not None:
            a = self.adapter_attn(a)
        x = x + self.drop(a)
        m = self.mlp(self.post_attention_layernorm(x))
        if self.adapter_mlp is not None:
            m = self.adapter_mlp(m)
        return x + self.drop(m), cache


class MakeAIModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.embed_proj = nn.Linear(cfg.emb_dim, cfg.hidden_size, bias=False) if cfg.emb_dim != cfg.hidden_size else None
        self.embed_positions = nn.Embedding(cfg.context_length, cfg.hidden_size) if cfg.pos_encoding == "learned" else None
        self.rotary = Rotary(cfg) if cfg.pos_encoding == "rope" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = make_norm(cfg)
        self.gradient_checkpointing = False

    def forward(self, input_ids, past=None, use_cache=False):
        B, T = input_ids.shape
        past_len = past[0][0].shape[2] if past else 0
        x = self.embed_tokens(input_ids)
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        if self.embed_positions is not None:
            pos = torch.arange(past_len, past_len + T, device=input_ids.device).clamp_max(self.cfg.context_length - 1)
            x = x + self.embed_positions(pos)[None]
        x = self.drop(x)
        cos = sin = None
        if self.rotary is not None:
            cos, sin = self.rotary.get(past_len + T, x.device, x.dtype)
        caches = []
        for i, layer in enumerate(self.layers):
            lp = past[i] if past else None
            if self.gradient_checkpointing and self.training:
                x, c = checkpoint(layer, x, cos, sin, lp, use_cache, use_reentrant=False)
            else:
                x, c = layer(x, cos, sin, lp, use_cache)
            caches.append(c)
        return self.norm(x), (caches if use_cache else None)


class MakeAIForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        errs = cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.cfg = cfg
        self.model = MakeAIModel(cfg)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.emb_dim, bias=False) if cfg.emb_dim != cfg.hidden_size else None
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        scaled = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=scaled)
        if cfg.tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def set_gradient_checkpointing(self, on: bool) -> None:
        self.model.gradient_checkpointing = on

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) in seen or (trainable_only and not p.requires_grad):
                continue
            seen.add(id(p))
            n += p.numel()
        return n

    def forward(self, input_ids, labels=None, past=None, use_cache=False, last_only=False):
        """Returns (logits, loss, caches).

        With ``labels`` the output head runs only on positions that have a target (label != -100),
        and logits are not returned. For chat data, where prompts are masked out, this avoids
        materialising a [batch, seq, vocab] tensor - the largest activation for big vocabularies.
        """
        h, caches = self.model(input_ids, past, use_cache)
        if labels is not None:
            flat = h.reshape(-1, h.size(-1))
            tgt = labels.reshape(-1)
            keep = tgt != -100
            if not bool(keep.all()):
                flat, tgt = flat[keep], tgt[keep]
            if self.out_proj is not None:
                flat = self.out_proj(flat)
            logits = self.lm_head(flat)
            if tgt.numel() == 0:
                return None, logits.sum() * 0.0, caches
            return None, F.cross_entropy(logits.float(), tgt), caches
        if last_only:
            h = h[:, -1:]
        if self.out_proj is not None:
            h = self.out_proj(h)
        return self.lm_head(h), None, caches
