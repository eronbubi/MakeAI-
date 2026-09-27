"""Translate between Hugging Face causal-LM configs and MakeAI configs."""
from __future__ import annotations

from typing import Any

from .config import ModelConfig

SUPPORTED_HF = {"LlamaForCausalLM": "llama", "MistralForCausalLM": "mistral", "Qwen2ForCausalLM": "qwen2"}


def from_hf_config(hf: dict[str, Any]) -> tuple[ModelConfig, list[str]]:
    """Build a MakeAI config from a HF ``config.json``. Returns (config, warnings)."""
    warnings: list[str] = []
    archs = hf.get("architectures") or []
    arch = next((SUPPORTED_HF[a] for a in archs if a in SUPPORTED_HF), None) or hf.get("model_type")
    if arch not in ("llama", "mistral", "qwen2"):
        raise ValueError(f"Unsupported architecture {archs or hf.get('model_type')}; "
                         f"MakeAI runs {', '.join(SUPPORTED_HF)} natively")
    n_heads = hf["num_attention_heads"]
    hidden = hf["hidden_size"]
    head_dim = hf.get("head_dim") or hidden // n_heads
    rs = hf.get("rope_scaling") or {}
    rtype = (rs.get("rope_type") or rs.get("type") or "none") if rs else "none"
    scaling, factor = "none", 1.0
    if rtype == "linear":
        scaling, factor = "linear", float(rs.get("factor", 1.0))
    elif rtype == "dynamic":
        scaling, factor = "ntk", float(rs.get("factor", 1.0))
        warnings.append("dynamic RoPE scaling mapped to static NTK scaling")
    elif rtype not in ("none", "default"):
        warnings.append(f"RoPE scaling '{rtype}' is not implemented; positions beyond the original "
                        f"training length may degrade")
    window = 0
    if arch == "mistral" and hf.get("sliding_window"):
        window = int(hf["sliding_window"])
    if arch == "qwen2" and hf.get("use_sliding_window") and hf.get("sliding_window"):
        window = int(hf["sliding_window"])
    cfg = ModelConfig(
        vocab_size=hf["vocab_size"],
        n_layers=hf["num_hidden_layers"],
        hidden_size=hidden,
        intermediate_size=hf["intermediate_size"],
        n_heads=n_heads,
        n_kv_heads=hf.get("num_key_value_heads") or n_heads,
        head_dim=head_dim,
        context_length=int(hf.get("max_position_embeddings", 2048)),
        activation="swiglu",
        norm="rmsnorm",
        norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
        pos_encoding="rope",
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        rope_scaling=scaling,
        rope_scaling_factor=factor,
        attention_type="sliding_window" if window else "full",
        sliding_window=window,
        tie_weights=bool(hf.get("tie_word_embeddings", False)),
        bias=bool(hf.get("mlp_bias", False)),
        qkv_bias=(arch == "qwen2") or bool(hf.get("attention_bias", False)),
        bos_token_id=_first(hf.get("bos_token_id"), 1),
        eos_token_id=_first(hf.get("eos_token_id"), 2),
        pad_token_id=_first(hf.get("pad_token_id"), 0),
        source_architecture=arch,
    )
    if hf.get("hidden_act", "silu") != "silu":
        warnings.append(f"hidden_act {hf.get('hidden_act')} treated as SwiGLU")
    return cfg, warnings


def _first(v, default):
    if isinstance(v, list):
        return v[0] if v else default
    return default if v is None else int(v)


def to_hf_config(cfg: ModelConfig, torch_dtype: str = "float32") -> dict[str, Any]:
    arch = cfg.hf_compatible()
    if arch is None:
        raise ValueError("This architecture has no exact Hugging Face equivalent "
                         "(needs RMSNorm + SwiGLU + RoPE, no MLP bias, unfactorised embeddings, full attention)")
    d: dict[str, Any] = {
        "architectures": ["Qwen2ForCausalLM" if arch == "qwen2" else "LlamaForCausalLM"],
        "model_type": arch,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.context_length,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": cfg.tie_weights,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "torch_dtype": torch_dtype,
        "attention_dropout": 0.0,
    }
    if arch == "llama":
        d["attention_bias"] = False
        d["mlp_bias"] = False
    if cfg.rope_scaling != "none":
        d["rope_scaling"] = {"type": "linear" if cfg.rope_scaling == "linear" else "dynamic",
                             "rope_type": "linear" if cfg.rope_scaling == "linear" else "dynamic",
                             "factor": cfg.rope_scaling_factor}
    return d
