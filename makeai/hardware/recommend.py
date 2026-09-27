"""Hardware-aware recommendation engine.

Given the scanned hardware, a complexity level and (optionally) a dataset size
or base model, produce a full AUTO configuration: architecture, precision,
batch/accumulation, optimiser, learning rate, dataset requirements and
estimates for VRAM, RAM and duration. Every value is only a suggestion; the UI
lets the user switch to CUSTOM and edit any field.

Estimates are analytical. ``makeai.train.probe`` replaces them with measured
peak memory and tokens/s by running real training steps of the chosen config.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ..model.config import ModelConfig

COMPLEXITY = {
    1: {"label": "Basic", "params": 8e6, "ctx": 256, "vocab": 8192, "tokens_per_step": 32_768, "lora_rank": 8, "adapter": 32},
    4: {"label": "Standard", "params": 30e6, "ctx": 512, "vocab": 16384, "tokens_per_step": 65_536, "lora_rank": 16, "adapter": 64},
    16: {"label": "Advanced", "params": 120e6, "ctx": 1024, "vocab": 32000, "tokens_per_step": 131_072, "lora_rank": 32, "adapter": 128},
    64: {"label": "Extreme", "params": 350e6, "ctx": 2048, "vocab": 32000, "tokens_per_step": 262_144, "lora_rank": 64, "adapter": 256},
    256: {"label": "Research", "params": 1.3e9, "ctx": 4096, "vocab": 50304, "tokens_per_step": 524_288, "lora_rank": 128, "adapter": 512},
}

WIDTHS = [128, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2560]
OPT_STATE_BYTES = {"adamw": 8, "adam": 8, "adamw8bit": 2, "adafactor": 0.1, "lion": 4, "sgd": 4}
QUANT_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1.0, "nf4": 0.53, "fp4": 0.53, "int4": 0.53}


# ------------------------------------------------------------------ shapes
def shape_for_params(target: float, vocab: int, ctx: int, tie: bool = True) -> ModelConfig:
    """Pick a Llama-style shape (depth/width ratio ~ standard scaling practice) near ``target`` params."""
    best, best_err = None, float("inf")
    for h in WIDTHS:
        heads = max(1, h // 64)
        head_dim = h // heads
        # grouped-query attention for larger models: kv heads must divide the query heads
        kv = heads if heads < 8 else max(d for d in range(1, heads // 4 + 1) if heads % d == 0)
        inter = int(math.ceil(8 * h / 3 / 64) * 64)
        # depth that hits the target for this width
        per_layer = h * heads * head_dim * 2 + 2 * h * kv * head_dim + 3 * h * inter + 2 * h
        emb = vocab * h * (1 if tie else 2)
        layers = round((target - emb) / per_layer) if target > emb else 1
        layers = max(2, min(48, layers))
        # keep aspect ratio sensible: width/layers between ~16 and ~128
        ratio = h / layers
        cfg = ModelConfig(vocab_size=vocab, n_layers=layers, hidden_size=h, intermediate_size=inter, n_heads=heads,
                          n_kv_heads=kv, head_dim=head_dim, context_length=ctx, tie_weights=tie)
        # Width/depth ratio of well-behaved published models is ~32-100 (GPT-2 small 64, Llama-7B 128).
        # Very deep, narrow stacks train unstably at small batch sizes, so they are penalised.
        shape_pen = 0.0 if 32 <= ratio <= 110 else 0.6 * abs(math.log(ratio / (32 if ratio < 32 else 110)))
        err = abs(math.log(cfg.param_count() / target)) + shape_pen
        if err < best_err:
            best, best_err = cfg, err
    return best


# ------------------------------------------------------------------ memory
def estimate_memory(cfg: ModelConfig, *, method: str, precision: str, micro_batch: int, ctx: int,
                    optimizer: str = "adamw", grad_ckpt: bool = False, lora_rank: int = 16,
                    base_quant: str = "bf16", adapter_size: int = 64) -> dict[str, float]:
    """Estimated GPU memory in bytes for one training configuration."""
    n = cfg.param_count()
    h, L, i = cfg.hidden_size, cfg.n_layers, cfg.intermediate_size
    act_b = 4 if precision in ("fp32", "tf32") else 2
    trainable = n
    if method in ("from_scratch", "full"):
        weights = n * 4                                    # fp32 master weights
        grads = n * 4
        opt = n * OPT_STATE_BYTES.get(optimizer, 8)
    else:
        linear = cfg.n_layers * (cfg.hidden_size * (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim +
                                 cfg.n_heads * cfg.head_dim * cfg.hidden_size + 3 * cfg.hidden_size * cfg.intermediate_size)
        rest = n - linear
        qb = QUANT_BYTES.get(base_quant, 2)
        weights = linear * qb + rest * 2
        if method in ("lora", "qlora"):
            per_layer = lora_rank * (2 * h + (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim + cfg.n_heads * cfg.head_dim) \
                + lora_rank * 3 * (h + i)
            trainable = L * per_layer
        else:  # adapter
            trainable = L * 2 * (2 * h * adapter_size + adapter_size + h)
        weights += trainable * 4
        grads = trainable * 4
        opt = trainable * OPT_STATE_BYTES.get(optimizer, 8)
    tokens = micro_batch * ctx
    # Activations kept for backward per layer per token (flash/mem-efficient attention, SwiGLU MLP).
    per_tok_layer = act_b * (8 * h + 2 * (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim + 4 * i) + h  # + dropout masks
    if method == "qlora" or base_quant in ("nf4", "fp4", "int4", "int8"):
        per_tok_layer += act_b * (3 * h + i)                # dequantised weights / extra casts
    if grad_ckpt:
        acts = tokens * L * h * act_b + tokens * per_tok_layer
    else:
        acts = tokens * L * per_tok_layer
    logits = tokens * cfg.vocab_size * (act_b + 4 + 4)      # logits, fp32 upcast, grad
    overhead = 350 * 2**20                                   # CUDA context + allocator slack
    total = weights + grads + opt + acts + logits + overhead
    return {"weights": weights, "grads": grads, "optimizer": opt, "activations": acts, "logits": logits,
            "overhead": overhead, "total": total, "trainable_params": trainable}


def _lr_for(params: float) -> float:
    pts = [(1e7, 1e-3), (1.25e8, 6e-4), (3.5e8, 3e-4), (7.6e8, 2.5e-4), (1.3e9, 2e-4), (7e9, 1.2e-4)]
    if params <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if params <= x1:
            t = (math.log(params) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return float(f"{math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0))):.2e}")
    return pts[-1][1]


def _precision(dev: dict[str, Any]) -> str:
    if dev.get("type") == "cuda":
        return "bf16" if dev.get("bf16") else "fp16"
    if dev.get("type") == "mps":
        return "fp16"
    return "fp32"


def _achievable_flops(dev: dict[str, Any], bench: dict[str, Any] | None, precision: str) -> tuple[float, str]:
    """Sustained training FLOP/s. Measured matmul peak x typical model FLOP utilisation."""
    mfu = 0.45
    if bench and bench.get("tflops"):
        key = {"bf16": "bf16", "fp16": "fp16", "tf32": "tf32"}.get(precision, "fp32")
        peak = bench["tflops"].get(key) or max(bench["tflops"].values())
        return peak * 1e12 * mfu, f"measured {peak} TFLOPS x {mfu:.0%} utilisation"
    if dev.get("type") == "cuda":
        return 10e12 * mfu, "no benchmark yet - assumed 10 TFLOPS peak (run the benchmark for a measured value)"
    return 0.3e12, "CPU - assumed 0.3 TFLOPS"


def recommend(hw: dict[str, Any], complexity: int = 4, method: str = "from_scratch", *,
              dataset_tokens: int | None = None, base_config: ModelConfig | None = None,
              vocab_size: int | None = None, bench: dict[str, Any] | None = None,
              context_length: int | None = None) -> dict[str, Any]:
    if complexity not in COMPLEXITY:
        complexity = min(COMPLEXITY, key=lambda c: abs(c - complexity))
    level = COMPLEXITY[complexity]
    dev = hw.get("best_device") or {"type": "cpu"}
    vram = (dev.get("vram_mb") or 0) * 2**20
    ram_total = hw.get("ram", {}).get("total_mb", 8192) * 2**20
    other_used = 0
    if dev.get("type") == "cuda":
        g = next((x for x in hw.get("gpus", []) if x.get("index") == dev.get("index")), {})
        other_used = (g.get("vram_used_mb") or 0) * 2**20
    budget = (vram - other_used) * 0.92 - 192 * 2**20 if dev.get("type") == "cuda" else ram_total * 0.5
    precision = _precision(dev)
    notes: list[str] = []
    if other_used > 0.15 * vram:
        notes.append(f"Other applications currently use {other_used / 2**30:.1f} GB of VRAM; recommendations use the "
                     f"{(vram - other_used) / 2**30:.1f} GB that is free. Close them to train larger models.")

    if method in ("lora", "qlora", "adapter", "full") and base_config is None:
        raise ValueError(f"{method} needs a base model")

    # ------------------------------------------------------------ architecture
    if method == "from_scratch":
        vocab = vocab_size or level["vocab"]
        if not vocab_size and dataset_tokens:
            # a vocabulary much larger than the data supports leaves most embeddings untrained
            data_vocab = 4096 if dataset_tokens < 2e6 else 8192 if dataset_tokens < 2e7 else 16384 if dataset_tokens < 2e8 else 65536
            if data_vocab < vocab:
                vocab = data_vocab
                notes.append(f"Vocabulary reduced to {vocab:,} tokens to suit a {dataset_tokens:,}-token dataset.")
        ctx = context_length or level["ctx"]
        target = level["params"]
        if dataset_tokens:
            data_cap = dataset_tokens * 4 / 20          # <= 4 epochs at 20 tokens/param
            if data_cap < target:
                notes.append(f"Dataset has {dataset_tokens:,} tokens; a {target/1e6:.0f}M model would need "
                             f"~{target*20:,.0f} tokens (20 tokens/parameter). Model size reduced to fit the data "
                             f"and avoid overfitting.")
                target = max(1e6, data_cap)
        cfg = shape_for_params(target, vocab, ctx)
        optimizer, ckpt = "adamw", False
        while True:
            m = estimate_memory(cfg, method=method, precision=precision, micro_batch=1, ctx=ctx, optimizer=optimizer,
                                grad_ckpt=True)
            if m["total"] <= budget or cfg.param_count() < 2e6:
                break
            if optimizer == "adamw" and dev.get("type") == "cuda":
                optimizer = "adamw8bit"
                continue
            target *= 0.8
            cfg = shape_for_params(target, vocab, ctx)
        if cfg.param_count() < level["params"] * 0.75 and not (dataset_tokens and dataset_tokens * 4 / 20 < level["params"]):
            notes.append(f"{level['label']} targets ~{level['params']/1e6:.0f}M parameters but "
                         f"{dev.get('vram_mb', 0)/1024:.1f} GB VRAM fits ~{cfg.param_count()/1e6:.0f}M for training.")
        if optimizer == "adamw8bit":
            notes.append("8-bit AdamW selected to fit optimizer state into VRAM.")
    else:
        cfg = base_config
        ctx = min(context_length or level["ctx"], base_config.context_length)
        optimizer, ckpt = "adamw", False

    # ------------------------------------------------------------- method setup
    lora_rank = level["lora_rank"]
    base_quant = "bf16" if precision == "bf16" else ("fp16" if precision == "fp16" else "fp32")
    if method == "qlora":
        base_quant = "nf4"
    if method == "lora":
        m = estimate_memory(cfg, method="lora", precision=precision, micro_batch=1, ctx=ctx, grad_ckpt=True,
                            lora_rank=lora_rank, base_quant=base_quant)
        if m["total"] > budget and dev.get("type") == "cuda":
            notes.append("A 16-bit base model does not fit in VRAM for LoRA; QLoRA (NF4) is recommended.")
    if method == "full":
        for opt in ("adamw", "adamw8bit", "adafactor"):
            optimizer = opt
            m = estimate_memory(cfg, method="full", precision=precision, micro_batch=1, ctx=ctx, optimizer=opt,
                                grad_ckpt=True)
            if m["total"] <= budget:
                break
        if optimizer != "adamw":
            notes.append(f"{optimizer} chosen so full fine-tuning optimizer state fits in VRAM.")
        if m["total"] > budget:
            notes.append("Full fine-tuning of this model does not fit in VRAM even with gradient checkpointing and "
                         "8-bit/factored optimizer state - use LoRA or QLoRA.")

    # ----------------------------------------------------------- micro batch
    def mem(mb, ck):
        return estimate_memory(cfg, method=method, precision=precision, micro_batch=mb, ctx=ctx, optimizer=optimizer,
                               grad_ckpt=ck, lora_rank=lora_rank, base_quant=base_quant,
                               adapter_size=level["adapter"])["total"]

    micro, ckpt = 1, False
    for ck in (False, True):
        mb = 1
        while mb * 2 <= 64 and mem(mb * 2, ck) <= budget:
            mb *= 2
        if mem(mb, ck) <= budget:
            micro, ckpt = mb, ck
            if mb >= 4 or ck:
                break
    if ckpt:
        notes.append("Gradient checkpointing enabled: trades ~30% speed for much lower activation memory.")
    tokens_per_step_target = level["tokens_per_step"] if method == "from_scratch" else max(16_384, level["tokens_per_step"] // 4)
    accum = max(1, round(tokens_per_step_target / (micro * ctx)))
    tokens_per_step = micro * accum * ctx

    # ---------------------------------------------------------- optimisation
    n_params = cfg.param_count()
    if method == "from_scratch":
        lr = _lr_for(n_params)
        recommended_tokens = int(20 * n_params)
        min_tokens = int(2 * n_params)
    elif method == "full":
        lr, recommended_tokens, min_tokens = 2e-5, 0, 0
    elif method == "adapter":
        lr, recommended_tokens, min_tokens = 1e-3, 0, 0
    else:
        lr, recommended_tokens, min_tokens = 2e-4, 0, 0

    epochs = 1
    if dataset_tokens:
        if method == "from_scratch":
            epochs = max(1, min(4, math.ceil(recommended_tokens / dataset_tokens)))
            if dataset_tokens * 4 < recommended_tokens:
                notes.append(f"Dataset is {dataset_tokens:,} tokens; {recommended_tokens:,} recommended. "
                             f"Training is capped at 4 epochs to limit memorisation.")
        else:
            epochs = 3 if dataset_tokens < 5e6 else (2 if dataset_tokens < 5e7 else 1)
        total_tokens = epochs * dataset_tokens
    else:
        total_tokens = recommended_tokens or 10 * tokens_per_step * 100
    steps = max(1, math.ceil(total_tokens / tokens_per_step))
    warmup = int(min(2000, max(20, steps * 0.02))) if method == "from_scratch" else int(min(200, max(10, steps * 0.03)))

    # ------------------------------------------------------------- estimates
    m = estimate_memory(cfg, method=method, precision=precision, micro_batch=micro, ctx=ctx, optimizer=optimizer,
                        grad_ckpt=ckpt, lora_rank=lora_rank, base_quant=base_quant, adapter_size=level["adapter"])
    flops_tok = cfg.flops_per_token() if method in ("from_scratch", "full") else cfg.flops_per_token() * 2 / 3
    if ckpt:
        flops_tok *= 4 / 3
    rate, rate_note = _achievable_flops(dev, bench, precision)
    duration = total_tokens * flops_tok / rate
    ram_needed = (n_params * (2 if method in ("lora", "qlora", "adapter") else 4) + 1.5 * 2**30 +
                  min(dataset_tokens or 0, 5e8) * 4)
    if ram_needed > ram_total * 0.85:
        notes.append("Estimated RAM use is close to installed memory; close other applications before training.")

    return {
        "mode": "auto",
        "complexity": complexity,
        "complexity_label": level["label"],
        "method": method,
        "device": dev,
        "model": cfg.to_dict(),
        "training": {
            "precision": precision if method != "qlora" else precision,
            "base_quant": base_quant if method in ("lora", "qlora", "adapter") else None,
            "optimizer": optimizer,
            "scheduler": "cosine",
            "learning_rate": lr,
            "min_lr_ratio": 0.1,
            "warmup_steps": warmup,
            "weight_decay": 0.1 if method in ("from_scratch", "full") else 0.0,
            "grad_clip": 1.0,
            "betas": [0.9, 0.95] if method == "from_scratch" else [0.9, 0.999],
            "dropout": 0.0 if method == "from_scratch" and (dataset_tokens or 1e12) >= n_params * 5 else 0.1,
            "epochs": epochs,
            "max_steps": steps,
            "micro_batch_size": micro,
            "gradient_accumulation": accum,
            "batch_size": micro * accum,
            "context_length": ctx,
            "tokens_per_step": tokens_per_step,
            "gradient_checkpointing": ckpt,
            "eval_every": max(25, min(500, steps // 20)),
            "save_every": max(50, min(1000, steps // 10)),
            "lora": {"rank": lora_rank, "alpha": 2 * lora_rank, "dropout": 0.05, "bias": "none",
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     "double_quant": True} if method in ("lora", "qlora") else None,
            "adapter": {"bottleneck": level["adapter"]} if method == "adapter" else None,
            "dataloader_workers": 2,
            "seed": 1337,
        },
        "dataset": {"recommended_tokens": recommended_tokens, "minimum_tokens": min_tokens,
                    "dataset_tokens": dataset_tokens, "total_training_tokens": int(total_tokens)},
        "estimates": {
            "params": n_params,
            "trainable_params": int(m["trainable_params"]),
            "vram_bytes": int(m["total"]),
            "vram_breakdown": {k: int(v) for k, v in m.items() if k not in ("total", "trainable_params")},
            "vram_budget_bytes": int(budget),
            "ram_bytes": int(ram_needed),
            "duration_s": round(duration),
            "throughput_basis": rate_note,
            "tokens_per_s_est": round(rate / flops_tok),
            "fits": m["total"] <= budget,
        },
        "notes": notes,
    }
