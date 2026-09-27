"""Export an AI from My AIs.

Formats: safetensors (native + Hugging Face layout), GGUF (llama.cpp), PyTorch,
ONNX, LoRA/QLoRA adapter (PEFT layout), tokenizer, config, complete MakeAI
package. Every format carries the creator attribution in its own metadata
mechanism so the original creator survives any export/import round trip.
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .. import registry, store
from ..model.config import ModelConfig
from ..model.hf_compat import to_hf_config

FORMATS = {
    "package": "Complete MakeAI package (.makeai)",
    "safetensors": "Safetensors weights (MakeAI names = Hugging Face Llama/Qwen2 names)",
    "hf": "Hugging Face folder (config.json + model.safetensors + tokenizer)",
    "gguf": "GGUF for llama.cpp / Ollama / LM Studio",
    "pytorch": "PyTorch (.pt, loads with torch.load(weights_only=True))",
    "onnx": "ONNX (input_ids -> logits, onnxruntime)",
    "tensorrt": "TensorRT engine",
    "lora": "LoRA / QLoRA adapter (PEFT layout)",
    "tokenizer": "Tokenizer",
    "config": "Model config",
}


def availability(uid: str) -> dict[str, dict[str, Any]]:
    """Which formats can be exported for this AI, with the reason when not."""
    m = registry.load_manifest(uid)
    out: dict[str, dict[str, Any]] = {}
    has_cfg = (registry.model_dir(uid) / "config.json").exists()
    trained = m.get("runnable") and has_cfg
    cfg = registry.get_config(uid) if has_cfg else None
    tok_kind = None
    tdir = registry.tokenizer_dir(uid)
    if not tdir.exists() and m.get("base_model"):
        tdir = registry.tokenizer_dir(m["base_model"])
    if tdir.exists():
        from ..data.tokenizer import load_tokenizer
        try:
            tok_kind = _gguf_tokenizer_kind(load_tokenizer(tdir))
        except Exception:
            tok_kind = None
    for f in FORMATS:
        ok, why = True, ""
        if f in ("safetensors", "hf", "gguf", "pytorch", "onnx", "tensorrt") and not trained:
            ok, why = False, "the AI has no trained weights yet"
        if f == "hf" and ok and cfg and cfg.hf_compatible() is None:
            ok, why = False, "architecture has no exact Hugging Face equivalent (needs RMSNorm+SwiGLU+RoPE, no MLP bias)"
        if f == "gguf" and ok:
            if cfg.hf_compatible() is None:
                ok, why = False, "GGUF export needs a llama/qwen2-compatible architecture (RMSNorm+SwiGLU+RoPE)"
            elif tok_kind is None:
                ok, why = False, "GGUF export needs a byte-level BPE or SentencePiece(BPE) tokenizer"
        if f == "tensorrt":
            try:
                import tensorrt  # noqa: F401
            except Exception:
                ok, why = False, ("TensorRT is not installed in this environment (install NVIDIA TensorRT for your "
                                  "CUDA version, or build an engine from the ONNX export with trtexec)")
        if f == "lora" and not m.get("adapters"):
            ok, why = False, "this AI was not trained with LoRA/QLoRA/adapters"
        if f == "lora" and ok and m["adapters"][0]["kind"] == "adapter":
            ok, why = True, "bottleneck adapters export in MakeAI format (no PEFT equivalent)"
        if f in ("tokenizer",) and not tdir.exists():
            ok, why = False, "no tokenizer"
        if f == "config" and not has_cfg:
            ok, why = False, "no MakeAI config"
        out[f] = {"label": FORMATS[f], "available": ok, "reason": why}
    return out


def _out_dir(uid: str) -> Path:
    d = store.exports_dir() / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _full_state_dict(uid: str) -> tuple[dict, ModelConfig, dict]:
    """Full-precision weights with any LoRA adapter merged in."""
    import torch
    from safetensors.torch import load_file

    from ..model.peft import merge_lora_into_state_dict
    m = registry.load_manifest(uid)
    cfg = registry.get_config(uid)
    wp = registry.weights_path(uid)
    if wp.exists():
        return load_file(str(wp)), cfg, m
    base = m.get("base_model")
    sd = load_file(str(registry.weights_path(base)))
    ad = (m.get("adapters") or [None])[0]
    if ad and ad["kind"] in ("lora", "qlora"):
        asd = load_file(str(registry.model_dir(uid) / ad["path"]))
        sd = merge_lora_into_state_dict(sd, asd, ad["alpha"] / ad["rank"])
    elif ad and ad["kind"] == "adapter":
        raise ValueError("bottleneck adapters cannot be merged into plain weights; export as MakeAI package or adapter")
    return sd, cfg, m


def _tokenizer_dir(uid: str) -> Path:
    t = registry.tokenizer_dir(uid)
    if t.exists():
        return t
    m = registry.load_manifest(uid)
    return registry.tokenizer_dir(m["base_model"])


def _tokenizer_blob(uid: str) -> dict[str, str]:
    d = _tokenizer_dir(uid)
    out = {}
    for f in ("makeai_tokenizer.json", "tokenizer.json", "tokenizer_config.json"):
        if (d / f).exists():
            out[f] = (d / f).read_text(encoding="utf-8")
    for f in ("spm.model", "tokenizer.model"):
        if (d / f).exists():
            out[f + ".b64"] = base64.b64encode((d / f).read_bytes()).decode()
    return out


def restore_tokenizer_blob(blob: dict[str, str], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for k, v in blob.items():
        if k.endswith(".b64"):
            (dst / k[:-4]).write_bytes(base64.b64decode(v))
        else:
            (dst / k).write_text(v, encoding="utf-8")


def _meta(m: dict, cfg: ModelConfig) -> dict[str, str]:
    meta = registry.attribution_metadata(m)
    meta["makeai.config"] = json.dumps(cfg.to_dict())
    return meta


# ------------------------------------------------------------------ formats
def export(uid: str, fmt: str, options: dict | None = None) -> Path:
    options = options or {}
    av = availability(uid)
    if fmt not in av:
        raise ValueError(f"unknown format {fmt}")
    if not av[fmt]["available"]:
        raise ValueError(f"{FORMATS[fmt]} not available: {av[fmt]['reason']}")
    fn = {"package": export_package, "safetensors": export_safetensors, "hf": export_hf, "gguf": export_gguf,
          "pytorch": export_pytorch, "onnx": export_onnx, "lora": export_lora, "tokenizer": export_tokenizer,
          "config": export_config}[fmt]
    path = fn(uid, **options)
    m = registry.load_manifest(uid)
    m.setdefault("provenance", []).append({"event": "exported", "format": fmt, "file": Path(path).name,
                                           "at": store.now_iso()})
    registry.save_manifest(m)
    return path


def export_safetensors(uid: str, dtype: str = "keep") -> Path:
    import torch
    from safetensors.torch import save_file
    sd, cfg, m = _full_state_dict(uid)
    if dtype in ("bf16", "fp16", "fp32"):
        t = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        sd = {k: v.to(t) for k, v in sd.items()}
    meta = _meta(m, cfg)
    meta["makeai.tokenizer"] = json.dumps(_tokenizer_blob(uid))
    meta["format"] = "pt"
    out = _out_dir(uid) / f"{uid}.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out), metadata=meta)
    return out


def export_hf(uid: str, dtype: str = "bf16") -> Path:
    """Folder loadable with transformers.AutoModelForCausalLM.from_pretrained, zipped."""
    import torch
    from safetensors.torch import save_file
    sd, cfg, m = _full_state_dict(uid)
    t = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype, torch.bfloat16)
    sd = {k: v.to(t).contiguous() for k, v in sd.items()}
    if cfg.tie_weights:
        sd.pop("lm_head.weight", None)
    d = _out_dir(uid) / f"{uid}-hf"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    save_file(sd, str(d / "model.safetensors"), metadata={**registry.attribution_metadata(m), "format": "pt"})
    hf = to_hf_config(cfg, {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(t, "float32"))
    hf["makeai"] = registry.public_manifest(m)
    (d / "config.json").write_text(json.dumps(hf, indent=2), encoding="utf-8")
    (d / "generation_config.json").write_text(json.dumps({"bos_token_id": cfg.bos_token_id,
                                                          "eos_token_id": cfg.eos_token_id}, indent=2))
    _write_hf_tokenizer(uid, d)
    (d / "README.md").write_text(_model_card(m, cfg), encoding="utf-8")
    zp = shutil.make_archive(str(d), "zip", d)
    return Path(zp)


def _write_hf_tokenizer(uid: str, d: Path) -> None:
    from ..data.tokenizer import load_tokenizer
    tdir = _tokenizer_dir(uid)
    tok = load_tokenizer(tdir)
    if (tdir / "tokenizer.json").exists():
        shutil.copyfile(tdir / "tokenizer.json", d / "tokenizer.json")
    if (tdir / "tokenizer_config.json").exists():
        shutil.copyfile(tdir / "tokenizer_config.json", d / "tokenizer_config.json")
        return
    if tok.kind == "sentencepiece":
        shutil.copyfile(tdir / "spm.model", d / "tokenizer.model")
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast" if tok.kind != "sentencepiece" else "LlamaTokenizer",
           "chat_template": tok.chat_template, "model_max_length": registry.get_config(uid).context_length,
           "clean_up_tokenization_spaces": False,
           "additional_special_tokens": tok.extra_specials}
    for role in ("bos", "eos", "pad", "unk"):
        if tok.specials.get(role):
            cfg[f"{role}_token"] = tok.specials[role]
    (d / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _model_card(m: dict, cfg: ModelConfig) -> str:
    oc = m.get("original_creator") or m["creator"]
    return (f"# {m['name']} {m['version']}\n\n{m.get('description', '')}\n\n"
            f"- Created by: {m['creator']['name']} (@{m['creator']['username']})\n"
            f"- Originally created by: {oc['name']} (@{oc['username']})\n"
            f"- MakeAI ID: `{m['id']}`\n- Parameters: {cfg.param_count():,}\n- Context: {cfg.context_length}\n"
            f"- Made with MakeAI by Convergent\n")


def export_pytorch(uid: str) -> Path:
    import torch
    sd, cfg, m = _full_state_dict(uid)
    out = _out_dir(uid) / f"{uid}.pt"
    torch.save({"state_dict": {k: v.contiguous() for k, v in sd.items()}, "makeai_config": json.dumps(cfg.to_dict()),
                "makeai_manifest": json.dumps(registry.public_manifest(m)),
                "makeai_attribution": json.dumps(registry.attribution_metadata(m)),
                "makeai_tokenizer": json.dumps(_tokenizer_blob(uid))}, out)
    return out


def export_config(uid: str) -> Path:
    cfg = registry.get_config(uid)
    d = {"makeai": cfg.to_dict()}
    if cfg.hf_compatible():
        d["huggingface"] = to_hf_config(cfg)
    d["manifest"] = registry.public_manifest(registry.load_manifest(uid))
    out = _out_dir(uid) / f"{uid}.config.json"
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return out


def export_tokenizer(uid: str) -> Path:
    d = _out_dir(uid) / f"{uid}-tokenizer"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(_tokenizer_dir(uid), d)
    if not (d / "tokenizer_config.json").exists():
        _write_hf_tokenizer(uid, d)
    return Path(shutil.make_archive(str(d), "zip", d))


def export_lora(uid: str) -> Path:
    """PEFT-compatible LoRA adapter (adapter_model.safetensors + adapter_config.json)."""
    from safetensors.torch import load_file, save_file
    m = registry.load_manifest(uid)
    ad = m["adapters"][0]
    sd = load_file(str(registry.model_dir(uid) / ad["path"]))
    d = _out_dir(uid) / f"{uid}-{ad['kind']}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    meta = registry.attribution_metadata(m)
    meta["format"] = "pt"
    if ad["kind"] in ("lora", "qlora"):
        peft_sd = {}
        for k, v in sd.items():
            if k.endswith(".lora_A") or k.endswith(".lora_B"):
                peft_sd[f"base_model.model.{k}.weight"] = v.contiguous()
            elif k.endswith(".base.bias"):
                peft_sd["base_model.model." + k.replace(".base.bias", ".bias")] = v.contiguous()
        save_file(peft_sd, str(d / "adapter_model.safetensors"), metadata=meta)
        base = registry.load_manifest(m["base_model"]) if m.get("base_model") else {}
        cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": ad["rank"], "lora_alpha": ad["alpha"],
               "lora_dropout": 0.0, "target_modules": ad.get("target_modules"), "bias": ad.get("bias", "none"),
               "base_model_name_or_path": base.get("hf_source") or base.get("id") or "", "inference_mode": True,
               "fan_in_fan_out": False, "init_lora_weights": True, "use_rslora": False, "use_dora": False}
        (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        # PEFT rejects unknown config keys, so attribution lives in its own file (and in the weights' metadata)
        (d / "makeai_manifest.json").write_text(json.dumps({**registry.public_manifest(m),
                                                            "base_quantization": ad.get("base_quant")}, indent=2),
                                                encoding="utf-8")
    else:
        save_file(sd, str(d / "adapter.safetensors"), metadata=meta)
        (d / "adapter_config.json").write_text(json.dumps({"makeai_adapter": ad, "makeai": registry.public_manifest(m)},
                                                          indent=2), encoding="utf-8")
    return Path(shutil.make_archive(str(d), "zip", d))


def export_package(uid: str, include_base: bool = True) -> Path:
    """.makeai = zip(manifest.json, config.json, tokenizer/, weights/, adapters/[, base/])."""
    m = registry.load_manifest(uid)
    src = registry.model_dir(uid)
    out = _out_dir(uid) / f"{uid}.makeai"
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("makeai-package.json", json.dumps({"package_version": 1, "id": m["id"]}))
        for p in src.rglob("*"):
            if p.is_file() and "exports" not in p.parts:
                z.write(p, p.relative_to(src).as_posix())
        if include_base and m.get("base_model") and not registry.weights_path(uid).exists():
            bdir = registry.model_dir(m["base_model"])
            for p in bdir.rglob("*"):
                if p.is_file() and "exports" not in p.parts:
                    z.write(p, "base/" + p.relative_to(bdir).as_posix())
    tmp.replace(out)
    return out


def export_onnx(uid: str, opset: int = 17, verify: bool = True) -> Path:
    import onnx
    import torch

    from ..registry import build_with_weights
    sd, cfg, m = _full_state_dict(uid)
    model = build_with_weights(cfg, sd, "cpu", torch.float32).eval()
    if model.model.rotary is not None:       # bake the RoPE table for the whole context into the graph
        model.model.rotary.get(cfg.context_length, torch.device("cpu"), torch.float32)

    class Wrapper(torch.nn.Module):
        def __init__(self, mdl):
            super().__init__()
            self.m = mdl

        def forward(self, input_ids):
            return self.m(input_ids)[0]

    out = _out_dir(uid) / f"{uid}.onnx"
    x = torch.randint(0, cfg.vocab_size, (1, min(16, cfg.context_length)))
    big = cfg.param_count() * 4 > 1.8 * 2**30
    torch.onnx.export(Wrapper(model), (x,), str(out), input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
                      opset_version=opset, do_constant_folding=True)
    mp = onnx.load(str(out), load_external_data=True)
    meta = _meta(m, cfg)
    meta["makeai.tokenizer"] = json.dumps(_tokenizer_blob(uid))
    for k, v in meta.items():
        e = mp.metadata_props.add()
        e.key, e.value = k, v
    if big:
        onnx.save_model(mp, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=out.name + ".data")
    else:
        onnx.save_model(mp, str(out))
    if verify:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        y = torch.randint(0, cfg.vocab_size, (1, min(24, cfg.context_length)))
        ref = model(y)[0].detach().numpy()
        got = sess.run(None, {"input_ids": y.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        if diff > 1e-2:
            raise RuntimeError(f"ONNX verification failed: max |logit diff| = {diff}")
    return out


# ------------------------------------------------------------------ GGUF
def _gguf_tokenizer_kind(tok) -> str | None:
    if tok.kind == "sentencepiece":
        mt = (tok.meta or {}).get("model_type")
        if mt == "bpe" or mt is None:
            try:
                from sentencepiece import sentencepiece_model_pb2 as pb
                p = pb.ModelProto()
                p.ParseFromString(open(tok.meta["spm_path"], "rb").read())
                return "llama" if p.trainer_spec.model_type == 2 else None
            except Exception:
                return "llama" if mt == "bpe" else None
        return None
    js = json.loads(tok.backend.to_str())
    model = js.get("model", {})
    pre = js.get("pre_tokenizer") or {}
    pres = [pre] + list(pre.get("pretokenizers", []) if isinstance(pre, dict) else [])
    if model.get("type") == "BPE" and any(p and p.get("type") == "ByteLevel" for p in pres):
        return "gpt2"
    return None


def _permute_qk(w, n_head: int):
    """llama.cpp's llama architecture uses interleaved RoPE; HF-layout q/k rows must be permuted."""
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape))


def export_gguf(uid: str, outtype: str = "f16") -> Path:
    import gguf
    import numpy as np
    import torch

    from ..data.tokenizer import load_tokenizer
    sd, cfg, m = _full_state_dict(uid)
    tok = load_tokenizer(_tokenizer_dir(uid))
    tkind = _gguf_tokenizer_kind(tok)
    arch = cfg.hf_compatible()
    out = _out_dir(uid) / f"{uid}.{outtype}.gguf"
    w = gguf.GGUFWriter(str(out), arch)
    oc = m.get("original_creator") or m["creator"]
    w.add_name(m["name"])
    w.add_author(f"{oc['name']} (@{oc['username']})")
    w.add_version(str(m["version"]))
    w.add_description(m.get("description") or "")
    w.add_organization("MakeAI by Convergent")
    for k, v in _meta(m, cfg).items():
        w.add_string(k, v)
    w.add_context_length(cfg.context_length)
    w.add_embedding_length(cfg.hidden_size)
    w.add_block_count(cfg.n_layers)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(cfg.n_kv_heads)
    w.add_key_length(cfg.head_dim)
    w.add_value_length(cfg.head_dim)
    w.add_rope_dimension_count(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    if cfg.rope_scaling == "linear":
        w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        w.add_rope_scaling_factor(cfg.rope_scaling_factor)
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if outtype == "f16" else gguf.LlamaFileType.ALL_F32)
    # tokenizer
    vocab = tok.vocab_size
    if cfg.vocab_size < vocab:
        raise ValueError(f"tokenizer has {vocab} tokens but the model only {cfg.vocab_size}")
    specials = tok.all_special_ids()
    if tkind == "gpt2":
        js = json.loads(tok.backend.to_str())
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("qwen2" if cfg.source_architecture == "qwen2" else "gpt-2")
        tokens = [tok.id_to_token(i) for i in range(vocab)]
        types = [gguf.TokenType.CONTROL if i in specials else gguf.TokenType.NORMAL for i in range(vocab)]
        added = {a["id"]: a for a in js.get("added_tokens", [])}
        for i, a in added.items():
            if i < vocab:
                types[i] = gguf.TokenType.CONTROL if a.get("special") else gguf.TokenType.USER_DEFINED
        merges = js["model"]["merges"]
        merges = [" ".join(x) if isinstance(x, list) else x for x in merges]
        w.add_token_merges(merges)
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tok.meta["spm_path"])
        w.add_tokenizer_model("llama")
        w.add_tokenizer_pre("default")
        tokens, scores, types = [], [], []
        for i in range(sp.get_piece_size()):
            tokens.append(sp.id_to_piece(i))
            scores.append(sp.get_score(i))
            if sp.is_unknown(i):
                types.append(gguf.TokenType.UNKNOWN)
            elif sp.is_control(i):
                types.append(gguf.TokenType.CONTROL)
            elif sp.is_byte(i):
                types.append(gguf.TokenType.BYTE)
            elif i in specials:
                types.append(gguf.TokenType.USER_DEFINED)
            else:
                types.append(gguf.TokenType.NORMAL)
    # pad the vocabulary if the embedding matrix is larger (e.g. Qwen pads to a multiple of 128)
    for i in range(len(tokens), cfg.vocab_size):
        tokens.append(f"[PAD{i}]")
        types.append(gguf.TokenType.UNUSED)
        if tkind != "gpt2":
            scores.append(-1000.0)
    if tkind != "gpt2":
        w.add_token_scores(scores)
    w.add_token_list(tokens)
    w.add_token_types(types)
    for role, fn in (("bos", w.add_bos_token_id), ("eos", w.add_eos_token_id), ("pad", w.add_pad_token_id),
                     ("unk", w.add_unk_token_id)):
        i = tok.special_id(role)
        if i is not None:
            fn(i)
    w.add_add_bos_token(tok.kind != "custom" and tok.bos_id is not None)
    if tok.chat_template:
        w.add_chat_template(tok.chat_template)
    eot = tok.token_to_id("<|end|>") or tok.token_to_id("<|im_end|>")
    if eot is not None:
        w.add_eot_token_id(eot)
    # tensors
    name_map = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
                "lm_head.weight": "output.weight"}
    per = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
           "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
           "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
           "self_attn.q_proj.bias": "attn_q.bias", "self_attn.k_proj.bias": "attn_k.bias",
           "self_attn.v_proj.bias": "attn_v.bias",
           "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
           "mlp.down_proj.weight": "ffn_down.weight"}
    if cfg.tie_weights:
        sd.pop("lm_head.weight", None)
    for k, v in sd.items():
        if k in name_map:
            name = name_map[k]
        else:
            parts = k.split(".")
            if parts[:2] != ["model", "layers"]:
                raise ValueError(f"unexpected tensor {k}")
            name = f"blk.{parts[2]}." + per[".".join(parts[3:])]
        t = v.float()
        if arch == "llama" and k.endswith("q_proj.weight"):
            t = _permute_qk(t, cfg.n_heads)
        elif arch == "llama" and k.endswith("k_proj.weight"):
            t = _permute_qk(t, cfg.n_kv_heads)
        a = t.numpy()
        if outtype == "f16" and a.ndim == 2:
            a = a.astype(np.float16)
        else:
            a = a.astype(np.float32)
        w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out
