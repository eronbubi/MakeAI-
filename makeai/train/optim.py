"""Optimisers and learning-rate schedules."""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch.optim import Optimizer

OPTIMIZERS = ("adam", "adamw", "adamw8bit", "adafactor", "lion", "sgd")
SCHEDULERS = ("constant", "linear", "cosine", "cosine_restarts", "polynomial", "one_cycle")


class Lion(Optimizer):
    """Lion (Chen et al., 2023): sign of an interpolated momentum, decoupled weight decay."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                update = m.mul(b1).add(p.grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


class Adafactor(Optimizer):
    """Adafactor (Shazeer & Stern, 2018) with factored second moments, explicit LR, no momentum."""

    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8,
                 weight_decay=0.0, scale_parameter=False):
        super().__init__(params, dict(lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
                                      weight_decay=weight_decay, scale_parameter=scale_parameter))

    @staticmethod
    def _rms(t):
        return t.norm(2) / (t.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                st = self.state[p]
                factored = grad.dim() >= 2
                if not st:
                    st["step"] = 0
                    if factored:
                        st["row"] = torch.zeros(grad.shape[:-1], device=grad.device)
                        st["col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device)
                    else:
                        st["sq"] = torch.zeros_like(grad)
                st["step"] += 1
                beta2 = 1.0 - st["step"] ** g["decay_rate"]
                upd = grad.pow(2).add_(g["eps"][0])
                if factored:
                    st["row"].mul_(beta2).add_(upd.mean(dim=-1), alpha=1 - beta2)
                    st["col"].mul_(beta2).add_(upd.mean(dim=-2), alpha=1 - beta2)
                    r = (st["row"] / st["row"].mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
                    c = st["col"].unsqueeze(-2).rsqrt()
                    u = grad * r * c
                else:
                    st["sq"].mul_(beta2).add_(upd, alpha=1 - beta2)
                    u = grad * st["sq"].rsqrt()
                u.div_((self._rms(u) / g["clip_threshold"]).clamp_(min=1.0))
                lr = g["lr"]
                if g["scale_parameter"]:
                    lr = lr * max(g["eps"][1], self._rms(p.float()).item())
                if g["weight_decay"]:
                    p.mul_(1 - g["weight_decay"] * lr)
                p.add_(u.to(p.dtype), alpha=-lr)
        return loss


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay on matrices only - not on norms, biases or embeddings."""
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.dim() < 2 or "norm" in name or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(name: str, model: torch.nn.Module, lr: float, weight_decay: float,
                    betas=(0.9, 0.95), momentum: float = 0.9) -> Optimizer:
    groups = param_groups(model, weight_decay)
    betas = tuple(betas)
    fused = all(p.is_cuda for g in groups for p in g["params"])
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    if name == "adam":
        for g in groups:
            g["weight_decay"] = 0.0 if g["weight_decay"] == 0 else g["weight_decay"]
        return torch.optim.Adam(groups, lr=lr, betas=betas, fused=fused)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except Exception as e:
            raise RuntimeError(f"AdamW 8-bit needs bitsandbytes with CUDA: {e}")
        return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas)
    if name == "adafactor":
        return Adafactor(groups, lr=lr)
    if name == "lion":
        return Lion(groups, lr=lr, betas=(betas[0], 0.99))
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=momentum > 0)
    raise ValueError(f"unknown optimizer {name}; choose from {OPTIMIZERS}")


def lr_lambda(kind: str, total: int, warmup: int, min_ratio: float = 0.1, cycles: int = 3, power: float = 2.0):
    """Multiplier on the base LR for step s (0-based). Warmup is linear for every schedule."""
    total = max(1, total)
    warmup = max(0, min(warmup, total - 1))

    def f(s: int) -> float:
        if kind == "one_cycle":
            up = max(1, warmup or int(0.3 * total))
            if s < up:
                return 0.04 + 0.96 * s / up
            p = min(1.0, (s - up) / max(1, total - up))
            return 1e-4 + (1 - 1e-4) * 0.5 * (1 + math.cos(math.pi * p))
        if warmup and s < warmup:
            return (s + 1) / warmup
        p = min(1.0, (s - warmup) / max(1, total - warmup))
        if kind == "constant":
            return 1.0
        if kind == "linear":
            return min_ratio + (1 - min_ratio) * (1 - p)
        if kind == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
        if kind == "cosine_restarts":
            q = (p * cycles) % 1.0 if p < 1 else 1.0
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * q))
        if kind == "polynomial":
            return min_ratio + (1 - min_ratio) * (1 - p) ** power
        raise ValueError(f"unknown scheduler {kind}; choose from {SCHEDULERS}")
    return f


def build_scheduler(opt: Optimizer, kind: str, total: int, warmup: int, min_ratio: float = 0.1, **kw):
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(kind, total, warmup, min_ratio, **kw))
