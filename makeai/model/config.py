"""Model architecture description for MakeAI's decoder-only transformer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

ACTIVATIONS = ("swiglu", "geglu", "gelu", "relu", "silu")
GATED = ("swiglu", "geglu")
NORMS = ("rmsnorm", "layernorm")
POSITIONAL = ("rope", "learned", "alibi", "none")
ROPE_SCALING = ("none", "linear", "ntk")
ATTENTION_TYPES = ("full", "sliding_window")


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 2048
    n_heads: int = 12
    n_kv_heads: int = 12
    head_dim: int = 64
    context_length: int = 1024
    embedding_size: int = 0          # 0 = same as hidden_size (no factorisation)
    activation: str = "swiglu"
    norm: str = "rmsnorm"
    norm_eps: float = 1e-5
    pos_encoding: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling: str = "none"
    rope_scaling_factor: float = 1.0
    attention_type: str = "full"
    sliding_window: int = 0
    tie_weights: bool = True
    bias: bool = False               # bias on o_proj / MLP / norms-free linears
    qkv_bias: bool = False           # Qwen2-style bias on q/k/v only
    dropout: float = 0.0
    attention_dropout: float = 0.0
    init_std: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    # Where the architecture came from when imported (e.g. "qwen2", "llama").
    source_architecture: str = "makeai"
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def emb_dim(self) -> int:
        return self.embedding_size or self.hidden_size

    @property
    def attention_kind(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "MHA"
        if self.n_kv_heads == 1:
            return "MQA"
        return "GQA"

    def validate(self) -> list[str]:
        errs = []
        if self.activation not in ACTIVATIONS: errs.append(f"activation must be one of {ACTIVATIONS}")
        if self.norm not in NORMS: errs.append(f"norm must be one of {NORMS}")
        if self.pos_encoding not in POSITIONAL: errs.append(f"pos_encoding must be one of {POSITIONAL}")
        if self.rope_scaling not in ROPE_SCALING: errs.append(f"rope_scaling must be one of {ROPE_SCALING}")
        if self.attention_type not in ATTENTION_TYPES: errs.append(f"attention_type must be one of {ATTENTION_TYPES}")
        for name in ("vocab_size", "n_layers", "hidden_size", "intermediate_size", "n_heads", "n_kv_heads", "head_dim", "context_length"):
            if int(getattr(self, name)) <= 0:
                errs.append(f"{name} must be positive")
        if self.n_kv_heads > 0 and self.n_heads % self.n_kv_heads:
            errs.append("n_heads must be divisible by n_kv_heads")
        if self.pos_encoding == "rope" and self.head_dim % 2:
            errs.append("head_dim must be even for RoPE")
        if self.attention_type == "sliding_window" and self.sliding_window <= 0:
            errs.append("sliding_window must be > 0 for sliding-window attention")
        if not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1:
            errs.append("dropout must be in [0, 1)")
        return errs

    def param_count(self) -> int:
        """Exact number of parameters of :class:`MakeAIForCausalLM` with this config."""
        h, e, v = self.hidden_size, self.emb_dim, self.vocab_size
        q = self.n_heads * self.head_dim
        kv = self.n_kv_heads * self.head_dim
        norm_p = h * (2 if self.norm == "layernorm" else 1)
        attn = h * q + 2 * h * kv + q * h
        if self.qkv_bias:
            attn += q + 2 * kv
        if self.bias:
            attn += h
        if self.activation in GATED:
            mlp = 3 * h * self.intermediate_size
            if self.bias:
                mlp += 2 * self.intermediate_size + h
        else:
            mlp = 2 * h * self.intermediate_size
            if self.bias:
                mlp += self.intermediate_size + h
        per_layer = attn + mlp + 2 * norm_p
        total = self.n_layers * per_layer + norm_p + v * e
        if e != h:
            total += 2 * e * h                     # in/out projections
        if not self.tie_weights:
            total += v * e
        if self.pos_encoding == "learned":
            total += self.context_length * h
        return total

    def flops_per_token(self) -> float:
        """Training FLOPs per token (forward + backward), 6N + attention term."""
        # The input embedding is a lookup (no FLOPs); the output head is a matmul.
        matmul_params = self.param_count() - (0 if self.tie_weights else self.vocab_size * self.emb_dim)
        if self.pos_encoding == "learned":
            matmul_params -= self.context_length * self.hidden_size
        attn = 6 * self.n_layers * self.n_heads * self.head_dim * self.context_length  # causal ~ half of 12*L*d*T
        return 6.0 * matmul_params + attn

    def hf_compatible(self) -> str | None:
        """Name of the Hugging Face architecture these weights map onto 1:1, if any."""
        if (self.norm == "rmsnorm" and self.activation == "swiglu" and self.pos_encoding == "rope"
                and self.emb_dim == self.hidden_size and not self.bias
                and self.attention_type == "full"):
            return "qwen2" if self.qkv_bias else "llama"
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["param_count"] = self.param_count()
        d["attention_kind"] = self.attention_kind
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
