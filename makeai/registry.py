"""My AIs: model registry, manifests, AI IDs and creator attribution.

Layout of one AI: ``models/<uid>/``
    manifest.json        identity, creator attribution, provenance, sharing
    config.json          MakeAI architecture (ModelConfig)
    tokenizer/           tokenizer files
    weights/model.safetensors   full weights (native or HF-compatible names)
    adapters/<name>/     LoRA / adapter weights when trained on a base model

Creator attribution is written into the manifest *and* into the metadata of
every weight file MakeAI writes, so it survives export and re-import.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from . import store
from .model.config import ModelConfig

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,38}$")


def normalize_username(u: str) -> str:
    u = (u or "").strip().lstrip("@").lower()
    if not USERNAME_RE.match(u):
        raise ValueError("username must be 1-39 characters: a-z, 0-9, '_', '-', '.'")
    return u


def ai_id(username: str, name: str, version: str) -> str:
    return f"makeai://{normalize_username(username)}/{store.slugify(name)}/{version}"


def uid_for(username: str, name: str, version: str) -> str:
    return f"{normalize_username(username)}--{store.slugify(name)}--{store.slugify(str(version))}"


def parse_ai_id(s: str) -> dict[str, str]:
    m = re.match(r"^makeai://([^/]+)/([^/]+)/([^/]+)$", s or "")
    if not m:
        raise ValueError(f"not a MakeAI ID: {s}")
    return {"username": m.group(1), "slug": m.group(2), "version": m.group(3)}


def model_dir(uid: str) -> Path:
    d = store.models_dir() / uid
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"AI {uid} not found")
    return d


def load_manifest(uid: str) -> dict[str, Any]:
    return store.read_json(model_dir(uid) / "manifest.json")


def save_manifest(m: dict[str, Any]) -> None:
    m["updated"] = store.now_iso()
    store.write_json(store.models_dir() / m["uid"] / "manifest.json", m)


def list_models() -> list[dict[str, Any]]:
    out = []
    for d in sorted(store.models_dir().iterdir()):
        m = store.read_json(d / "manifest.json")
        if m:
            out.append(m)
    out.sort(key=lambda m: m.get("updated", ""), reverse=True)
    return out


def attribution_metadata(m: dict[str, Any]) -> dict[str, str]:
    """Flat string metadata embedded in weight files (safetensors / GGUF / ONNX)."""
    oc = m.get("original_creator") or m["creator"]
    return {
        "makeai.id": m["id"],
        "makeai.name": m["name"],
        "makeai.version": str(m["version"]),
        "makeai.creator.name": m["creator"]["name"],
        "makeai.creator.username": m["creator"]["username"],
        "makeai.original_creator.name": oc["name"],
        "makeai.original_creator.username": oc["username"],
        "makeai.description": m.get("description", ""),
        "makeai.created": m.get("created", ""),
        "makeai.manifest": json.dumps(public_manifest(m), ensure_ascii=False),
    }


def public_manifest(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "name", "version", "description", "tags", "icon", "creator", "original_creator", "imported_by",
            "created", "updated", "param_count", "context_length", "format", "architecture_summary",
            "required_hardware", "provenance", "method", "base_model", "license")
    return {k: m.get(k) for k in keys if k in m}


def architecture_summary(cfg: ModelConfig) -> dict[str, Any]:
    return {"layers": cfg.n_layers, "hidden": cfg.hidden_size, "heads": cfg.n_heads, "kv_heads": cfg.n_kv_heads,
            "attention": cfg.attention_kind, "activation": cfg.activation, "norm": cfg.norm,
            "positional": cfg.pos_encoding, "vocab": cfg.vocab_size, "context": cfg.context_length,
            "source": cfg.source_architecture}


def required_hardware(cfg: ModelConfig) -> dict[str, Any]:
    n = cfg.param_count()
    kv_per_tok = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 2
    return {"inference_bf16_gb": round((n * 2 + kv_per_tok * cfg.context_length) / 2**30 + 0.4, 2),
            "inference_int8_gb": round((n * 1 + kv_per_tok * cfg.context_length) / 2**30 + 0.4, 2),
            "inference_nf4_gb": round((n * 0.55 + kv_per_tok * cfg.context_length) / 2**30 + 0.4, 2),
            "full_training_gb": round(n * 16 / 2**30 + 1, 1)}


def create_model(*, name: str, creator_name: str, creator_username: str, description: str = "",
                 version: str = "1.0", tags: list[str] | None = None, icon: str = "",
                 config: ModelConfig | None = None, method: str = "from_scratch",
                 base_model: str | None = None, license: str = "") -> dict[str, Any]:
    if not name.strip():
        raise ValueError("AI name is required")
    if not creator_name.strip():
        raise ValueError("creator name is required")
    username = normalize_username(creator_username)
    uid = uid_for(username, name, version)
    d = store.models_dir() / uid
    if (d / "manifest.json").exists():
        raise FileExistsError(f"{ai_id(username, name, version)} already exists - choose another version")
    d.mkdir(parents=True, exist_ok=True)
    creator = {"name": creator_name.strip(), "username": username}
    m: dict[str, Any] = {
        "uid": uid,
        "id": ai_id(username, name, version),
        "name": name.strip(),
        "version": str(version),
        "description": description,
        "tags": [t.strip() for t in (tags or []) if t.strip()],
        "icon": icon,
        "license": license,
        "creator": creator,
        "original_creator": dict(creator),
        "imported_by": [],
        "created": store.now_iso(),
        "status": "untrained" if method == "from_scratch" else "configured",
        "method": method,
        "base_model": base_model,
        "format": "makeai",
        "runnable": False,
        "provenance": [{"event": "created", "by": username, "at": store.now_iso(), "method": method}],
        "sharing": {"visibility": "private", "token": None},
        "downloads": 0,
        "training_runs": [],
        "evaluations": [],
    }
    if config is not None:
        store.write_json(d / "config.json", config.to_dict())
        m["param_count"] = config.param_count()
        m["context_length"] = config.context_length
        m["architecture_summary"] = architecture_summary(config)
        m["required_hardware"] = required_hardware(config)
    save_manifest(m)
    return m


def update_model(uid: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Edit descriptive fields. Creator and original creator can never be replaced."""
    m = load_manifest(uid)
    for k in ("description", "tags", "icon", "license"):
        if k in patch:
            m[k] = patch[k]
    if "name" in patch and patch["name"] and patch["name"] != m["name"]:
        m["display_name"] = patch["name"]  # the ID stays stable; display name may change
    m.setdefault("provenance", []).append({"event": "edited", "at": store.now_iso(),
                                           "fields": sorted(k for k in patch if k not in ("creator", "original_creator"))})
    save_manifest(m)
    return m


def new_version(uid: str, version: str, by_username: str) -> dict[str, Any]:
    """Copy an AI to a new version. Keeps creator and original creator."""
    src = load_manifest(uid)
    new_uid = uid_for(src["creator"]["username"], src["name"], version)
    dst = store.models_dir() / new_uid
    if dst.exists():
        raise FileExistsError(f"version {version} already exists")
    shutil.copytree(model_dir(uid), dst, ignore=shutil.ignore_patterns("exports"))
    m = dict(src)
    m.update({"uid": new_uid, "version": str(version), "id": ai_id(src["creator"]["username"], src["name"], version),
              "created": store.now_iso(), "sharing": {"visibility": "private", "token": None}, "downloads": 0})
    m["provenance"] = list(src.get("provenance", [])) + [{"event": "new_version", "from": src["id"],
                                                          "by": by_username, "at": store.now_iso()}]
    save_manifest(m)
    return m


def settle_status(uid: str, run_state: str) -> None:
    """After a run ends without completing, the AI must not stay marked as 'training'."""
    try:
        m = load_manifest(uid)
    except FileNotFoundError:
        return
    if m.get("status") != "training":
        return
    if m.get("runnable"):
        m["status"] = "trained" if any(p.get("event") == "trained" for p in m.get("provenance", [])) else "partially_trained"
    else:
        m["status"] = "untrained" if m.get("method") == "from_scratch" else "configured"
    m.setdefault("provenance", []).append({"event": f"run_{run_state}", "at": store.now_iso()})
    save_manifest(m)


def delete_model(uid: str) -> None:
    shutil.rmtree(model_dir(uid))


def get_config(uid: str) -> ModelConfig:
    d = store.read_json(model_dir(uid) / "config.json")
    if not d:
        raise FileNotFoundError(f"{uid} has no MakeAI config (format: {load_manifest(uid).get('format')})")
    return ModelConfig.from_dict(d)


def tokenizer_dir(uid: str) -> Path:
    return model_dir(uid) / "tokenizer"


def weights_path(uid: str) -> Path:
    return model_dir(uid) / "weights" / "model.safetensors"


def save_weights(uid: str, state_dict: dict, extra_meta: dict[str, str] | None = None) -> Path:
    """Atomically write full weights with attribution metadata."""
    from safetensors.torch import save_file
    m = load_manifest(uid)
    meta = attribution_metadata(m)
    meta["format"] = "pt"
    meta.update(extra_meta or {})
    path = weights_path(uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    sd = _dedupe_shared(state_dict)
    save_file(sd, str(tmp), metadata=meta)
    tmp.replace(path)
    return path


def _dedupe_shared(sd: dict) -> dict:
    """safetensors refuses shared storage (tied embeddings); keep one copy, clone contiguous."""
    out, seen = {}, {}
    for k, v in sd.items():
        key = (v.data_ptr(), v.shape) if v.device.type != "meta" else None
        if key in seen and k == "lm_head.weight":
            continue
        seen[key] = k
        out[k] = v.detach().contiguous().cpu()
    return out


def adapter_dir(uid: str, name: str = "default") -> Path:
    return model_dir(uid) / "adapters" / name


# ------------------------------------------------------------------ loading
def build_with_weights(cfg: ModelConfig, sd: dict, device, dtype):
    """Create the model directly on ``device`` in ``dtype`` and load ``sd`` without a random-init pass."""
    import torch

    from .model.transformer import MakeAIForCausalLM, Rotary, alibi_slopes
    with torch.device("meta"):
        model = MakeAIForCausalLM(cfg)
    model = model.to(dtype).to_empty(device=device)
    sd = dict(sd)
    if cfg.tie_weights:
        sd.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not (cfg.tie_weights and k == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"weights do not match architecture: missing={missing[:4]} unexpected={unexpected[:4]}")
    if cfg.tie_weights:
        model.lm_head.weight = model.model.embed_tokens.weight
    if model.model.rotary is not None:          # non-persistent buffers were created on meta
        model.model.rotary = Rotary(cfg).to(device)
    for mod in model.modules():
        if hasattr(mod, "slopes"):
            mod.slopes = alibi_slopes(cfg.n_heads).to(device)
    return model


def load_model(uid: str, device: str = "cuda", dtype=None, quant: str | None = None, merge_adapters: bool = True):
    """Instantiate a runnable MakeAI model with its weights (and adapters) loaded.

    Returns (model, tokenizer, manifest). ``quant`` in {None, 'int8', 'nf4', 'fp4'}.
    """
    import torch
    from safetensors.torch import load_file

    from .data.tokenizer import load_tokenizer
    from .model.peft import (apply_adapters, apply_lora, load_trainable, merge_lora_into_state_dict,
                             quantize_linears)
    from .model.transformer import MakeAIForCausalLM

    m = load_manifest(uid)
    cfg = get_config(uid)
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
            torch.float16 if device.startswith("cuda") else torch.float32)
    base_uid = m.get("base_model") if m.get("method") in ("lora", "qlora", "adapter") else None
    wp = weights_path(uid)
    if not wp.exists() and base_uid:
        wp = weights_path(base_uid)
    if not wp.exists():
        raise FileNotFoundError(f"{m['name']} has no trained weights yet")
    sd = load_file(str(wp))
    adapters = m.get("adapters") or []
    active = next((a for a in adapters if a.get("active")), adapters[-1] if adapters else None)
    lora_sd = None
    if active and base_uid and not weights_path(uid).exists():
        asd = load_file(str(model_dir(uid) / active["path"]))
        if active["kind"] in ("lora", "qlora") and merge_adapters and quant is None:
            sd = merge_lora_into_state_dict(sd, asd, active["alpha"] / active["rank"])
        else:
            lora_sd = asd
    model = build_with_weights(cfg, sd, "cpu" if quant else device, dtype)
    if quant:
        quantize_linears(model, quant, compute_dtype=dtype, device=device)
        model.to(device)
    if lora_sd is not None:
        if active["kind"] == "adapter":
            apply_adapters(model, active["bottleneck"])
        else:
            apply_lora(model, active["rank"], active["alpha"], 0.0, active["target_modules"])
        load_trainable(model, lora_sd)
        from .model.transformer import BottleneckAdapter
        for mod in model.modules():            # adapters/LoRA were created in fp32; run them in the model dtype
            if isinstance(mod, BottleneckAdapter):
                mod.to(dtype)
        model.to(device)
    model.eval()
    tok = load_tokenizer(tokenizer_dir(uid)) if tokenizer_dir(uid).exists() else load_tokenizer(tokenizer_dir(base_uid))
    return model, tok, m
