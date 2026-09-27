"""Import models into My AIs, inspecting them and preserving the original creator."""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .. import registry, store
from ..model.config import ModelConfig
from ..model.hf_compat import from_hf_config

SUPPORTED = (".makeai", ".safetensors", ".gguf", ".pt", ".pth", ".onnx")


def _hf_org_from_path(p: Path) -> tuple[str | None, str | None]:
    m = re.search(r"models--([^\\/]+?)--([^\\/]+)", str(p))
    return (m.group(1), m.group(2)) if m else (None, None)


def _attr_from_flat(meta: dict[str, str]) -> dict[str, Any] | None:
    if not meta or "makeai.id" not in meta:
        return None
    manifest = {}
    try:
        manifest = json.loads(meta.get("makeai.manifest") or "{}")
    except json.JSONDecodeError:
        pass
    return {
        "id": meta["makeai.id"], "name": meta.get("makeai.name"), "version": meta.get("makeai.version"),
        "description": meta.get("makeai.description", ""),
        "creator": {"name": meta.get("makeai.creator.name"), "username": meta.get("makeai.creator.username")},
        "original_creator": {"name": meta.get("makeai.original_creator.name") or meta.get("makeai.creator.name"),
                             "username": meta.get("makeai.original_creator.username") or meta.get("makeai.creator.username")},
        "manifest": manifest,
    }


def _gguf_fields(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from gguf import GGUFReader, GGUFValueType
    r = GGUFReader(str(path))
    kv: dict[str, Any] = {}
    for name, f in r.fields.items():
        try:
            t = f.types[0] if f.types else None
            if t == GGUFValueType.STRING:
                kv[name] = bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace")
            elif t == GGUFValueType.ARRAY:
                kv[name] = f"<array of {len(f.data)}>"
            elif f.data:
                v = f.parts[f.data[0]]
                kv[name] = v.tolist()[0] if hasattr(v, "tolist") else v
        except Exception:
            continue
    tensors = [{"name": t.name, "shape": [int(x) for x in t.shape], "type": t.tensor_type.name,
                "elements": int(t.n_elements)} for t in r.tensors]
    return kv, tensors


def inspect(path: str) -> dict[str, Any]:
    """Describe a model file/folder without importing it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    info: dict[str, Any] = {"path": str(p), "runnable": False, "backend": None, "attribution": None, "warnings": []}
    suffix = p.suffix.lower()
    if p.is_dir():
        sts = sorted(p.glob("*.safetensors"))
        if (p / "manifest.json").exists():
            suffix = ".makeai-dir"
        elif sts:
            info.update(_inspect_safetensors(sts, p))
            return info
        else:
            raise ValueError("folder contains no .safetensors weights")
    if suffix == ".makeai":
        with zipfile.ZipFile(p) as z:
            man = json.loads(z.read("manifest.json"))
            cfg = json.loads(z.read("config.json")) if "config.json" in z.namelist() else None
            has_base = any(n.startswith("base/") for n in z.namelist())
        info.update({"format": "makeai-package", "name": man["name"], "version": man["version"], "id": man["id"],
                     "description": man.get("description", ""), "param_count": man.get("param_count"),
                     "context_length": man.get("context_length"), "runnable": bool(man.get("runnable")),
                     "backend": "native" if man.get("runnable") else None, "includes_base": has_base,
                     "attribution": {"creator": man["creator"], "original_creator": man.get("original_creator") or man["creator"],
                                     "imported_by": man.get("imported_by", [])},
                     "architecture": cfg})
    elif suffix == ".safetensors":
        info.update(_inspect_safetensors([p], p.parent))
    elif suffix == ".gguf":
        kv, tensors = _gguf_fields(p)
        n = sum(t["elements"] for t in tensors)
        arch = kv.get("general.architecture")
        attr = _attr_from_flat({k: v for k, v in kv.items() if isinstance(v, str)})
        try:
            import llama_cpp  # noqa: F401
            runnable, why = True, ""
        except Exception:
            runnable, why = False, "llama-cpp-python is not installed"
        if kv.get("general.type") == "model" and arch and "embed" in (kv.get("general.name") or "").lower():
            info["warnings"].append("this looks like an embedding model; chat output may be meaningless")
        qtypes = sorted({t["type"] for t in tensors})
        info.update({"format": "gguf", "architecture_name": arch, "name": kv.get("general.name") or p.stem,
                     "param_count": n, "context_length": kv.get(f"{arch}.context_length"),
                     "layers": kv.get(f"{arch}.block_count"), "hidden": kv.get(f"{arch}.embedding_length"),
                     "heads": kv.get(f"{arch}.attention.head_count"), "quantization": qtypes,
                     "author": kv.get("general.author"), "tensor_count": len(tensors),
                     "tokenizer": kv.get("tokenizer.ggml.model"), "runnable": runnable,
                     "backend": "llama.cpp" if runnable else None, "reason": why, "attribution": attr,
                     "metadata": {k: v for k, v in kv.items() if not k.startswith("tokenizer.ggml.") and
                                  not k.startswith("makeai.")}})
    elif suffix in (".pt", ".pth"):
        import torch
        obj = torch.load(str(p), map_location="cpu", weights_only=True, mmap=True)
        sd = obj.get("state_dict") or obj.get("model") or obj if isinstance(obj, dict) else None
        tensors = {k: v for k, v in (sd or {}).items() if hasattr(v, "shape")}
        n = sum(v.numel() for v in tensors.values())
        cfg = json.loads(obj["makeai_config"]) if isinstance(obj, dict) and "makeai_config" in obj else None
        attr = _attr_from_flat(json.loads(obj["makeai_attribution"])) if isinstance(obj, dict) and "makeai_attribution" in obj else None
        info.update({"format": "pytorch", "name": (attr or {}).get("name") or p.stem, "param_count": n,
                     "tensor_count": len(tensors), "architecture": cfg, "attribution": attr,
                     "runnable": cfg is not None and "makeai_tokenizer" in obj,
                     "backend": "native" if cfg is not None else None,
                     "reason": "" if cfg else "plain state dict without an architecture description - inspect only",
                     "sample_tensors": [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in list(tensors.items())[:12]]})
    elif suffix == ".onnx":
        import onnx
        mp = onnx.load(str(p), load_external_data=False)
        meta = {e.key: e.value for e in mp.metadata_props}
        attr = _attr_from_flat(meta)
        n = 0
        for t in mp.graph.initializer:
            k = 1
            for d in t.dims:
                k *= d
            n += k
        ins = [{"name": i.name, "shape": [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]} for i in mp.graph.input]
        outs = [{"name": o.name, "shape": [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]} for o in mp.graph.output]
        lm = any(i["name"] == "input_ids" for i in ins) and any(o["name"] == "logits" for o in outs)
        info.update({"format": "onnx", "name": (attr or {}).get("name") or mp.graph.name or p.stem, "param_count": n,
                     "opset": [o.version for o in mp.opset_import], "inputs": ins, "outputs": outs, "attribution": attr,
                     "producer": mp.producer_name, "runnable": lm and "makeai.tokenizer" in meta,
                     "backend": "onnxruntime" if lm and "makeai.tokenizer" in meta else None,
                     "reason": "" if lm and "makeai.tokenizer" in meta else
                     "not a causal-LM graph with an embedded tokenizer - inspect only",
                     "architecture": json.loads(meta["makeai.config"]) if "makeai.config" in meta else None})
    elif suffix != ".makeai-dir":
        raise ValueError(f"unsupported file type {suffix}; supported: {', '.join(SUPPORTED)}")
    return info


def _inspect_safetensors(files: list[Path], folder: Path) -> dict[str, Any]:
    from safetensors import safe_open
    n, count, meta, dtypes = 0, 0, {}, set()
    samples = []
    for f in files:
        with safe_open(str(f), "pt") as sf:
            meta.update(sf.metadata() or {})
            for k in sf.keys():
                sl = sf.get_slice(k)
                shape = sl.get_shape()
                c = 1
                for d in shape:
                    c *= d
                n += c
                count += 1
                dtypes.add(sl.get_dtype())
                if len(samples) < 12:
                    samples.append({"name": k, "shape": shape, "dtype": sl.get_dtype()})
    attr = _attr_from_flat(meta)
    info: dict[str, Any] = {"format": "safetensors", "param_count": n, "tensor_count": count, "dtypes": sorted(dtypes),
                            "attribution": attr, "sample_tensors": samples, "files": [f.name for f in files]}
    org, repo = _hf_org_from_path(folder)
    if "makeai.config" in meta:
        cfg = ModelConfig.from_dict(json.loads(meta["makeai.config"]))
        info.update({"architecture": cfg.to_dict(), "runnable": "makeai.tokenizer" in meta or (folder / "tokenizer.json").exists(),
                     "backend": "native", "name": attr["name"] if attr else files[0].stem})
    elif (folder / "config.json").exists():
        hf = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        try:
            cfg, warns = from_hf_config(hf)
            info.update({"architecture": cfg.to_dict(), "hf_architecture": hf.get("architectures"),
                         "runnable": (folder / "tokenizer.json").exists() or (folder / "tokenizer.model").exists(),
                         "backend": "native", "warnings": warns,
                         "name": (attr or {}).get("name") or (hf.get("makeai") or {}).get("name") or repo or folder.name})
            if cfg.param_count() != n:
                info["warnings"].append(f"parameter count from config ({cfg.param_count():,}) differs from tensors ({n:,})")
        except ValueError as e:
            info.update({"runnable": False, "reason": str(e), "name": repo or folder.name})
        info["hf_org"], info["hf_repo"] = org, repo
    else:
        info.update({"runnable": False, "reason": "no config.json or MakeAI metadata describing the architecture",
                     "name": files[0].stem})
    return info


# ------------------------------------------------------------------ import
def import_model(path: str, *, importer: dict[str, str], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Import into My AIs. ``importer`` = local profile {name, username} (recorded in imported_by).

    The original creator comes from the file's own metadata when present and is never replaced.
    For foreign files without metadata, ``overrides['creator_name'/'creator_username']`` names the
    original author (e.g. the organisation that published the weights).
    """
    o = overrides or {}
    info = inspect(path)
    p = Path(path)
    imp = {"name": importer.get("name") or "", "username": registry.normalize_username(importer["username"]),
           "at": store.now_iso()}
    if info["format"] == "makeai-package":
        return _import_package(p, imp, o)
    attr = info.get("attribution")
    if attr and attr.get("creator", {}).get("username"):
        creator = attr["creator"]
        original = attr["original_creator"]
        name = attr.get("name") or info.get("name")
        version = str(o.get("version") or attr.get("version") or "1.0")
        description = attr.get("description", "")
        prior_imports = (attr.get("manifest") or {}).get("imported_by") or []
    else:
        cu = o.get("creator_username") or info.get("hf_org") or ""
        cn = o.get("creator_name") or info.get("hf_org") or ""
        if not cu or not cn:
            raise ValueError("this file carries no creator information - enter the original creator's name and username")
        creator = {"name": cn, "username": registry.normalize_username(cu)}
        original = dict(creator)
        name = o.get("name") or info.get("name") or p.stem
        version = str(o.get("version") or "1.0")
        description = o.get("description") or ""
        prior_imports = []
    uid = registry.uid_for(creator["username"], name, version)
    d = store.models_dir() / uid
    if (d / "manifest.json").exists():
        raise FileExistsError(f"{registry.ai_id(creator['username'], name, version)} is already in My AIs")
    d.mkdir(parents=True)
    try:
        fmt = info["format"]
        cfg = ModelConfig.from_dict(info["architecture"]) if info.get("architecture") else None
        (d / "weights").mkdir()
        if fmt == "safetensors":
            _copy_safetensors(p, d, cfg, info)
        elif fmt == "gguf":
            shutil.copyfile(p, d / "weights" / "model.gguf")
        elif fmt == "pytorch":
            _import_pt(p, d)
        elif fmt == "onnx":
            shutil.copyfile(p, d / "weights" / "model.onnx")
            ext = p.with_name(p.name + ".data")
            if ext.exists():
                shutil.copyfile(ext, d / "weights" / ext.name)
            _onnx_tokenizer(d / "weights" / "model.onnx", d / "tokenizer")
        if cfg is not None:
            store.write_json(d / "config.json", cfg.to_dict())
        m = {
            "uid": uid, "id": registry.ai_id(creator["username"], name, version), "name": name, "version": version,
            "description": o.get("description") or description, "tags": o.get("tags") or [], "icon": o.get("icon", ""),
            "license": o.get("license", ""),
            "creator": creator, "original_creator": original,
            "imported_by": prior_imports + [imp],
            "created": store.now_iso(), "status": "imported", "method": "imported", "base_model": None,
            "format": {"safetensors": "makeai", "pytorch": "makeai", "gguf": "gguf", "onnx": "onnx"}[fmt],
            "source_format": fmt, "source_path": str(p),
            "runnable": bool(info.get("runnable")), "backend": info.get("backend") or None,
            "not_runnable_reason": info.get("reason", ""),
            "param_count": info.get("param_count"), "context_length": (cfg.context_length if cfg else info.get("context_length")),
            "provenance": [{"event": "imported", "by": imp["username"], "at": imp["at"], "from": fmt, "path": str(p)}],
            "sharing": {"visibility": "private", "token": None}, "downloads": 0, "training_runs": [], "evaluations": [],
        }
        if info.get("hf_org"):
            m["hf_source"] = f"{info['hf_org']}/{info['hf_repo']}"
        if cfg:
            m["architecture_summary"] = registry.architecture_summary(cfg)
            m["required_hardware"] = registry.required_hardware(cfg)
        elif fmt == "gguf":
            m["architecture_summary"] = {"source": info.get("architecture_name"), "layers": info.get("layers"),
                                         "hidden": info.get("hidden"), "heads": info.get("heads"),
                                         "quantization": info.get("quantization"), "context": info.get("context_length")}
        registry.save_manifest(m)
        return m
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def _copy_safetensors(p: Path, d: Path, cfg: ModelConfig | None, info: dict) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file
    folder = p if p.is_dir() else p.parent
    files = [folder / f for f in info["files"]] if p.is_dir() else [p]
    meta: dict[str, str] = {}
    if len(files) == 1:
        shutil.copyfile(files[0], d / "weights" / "model.safetensors")
        with safe_open(str(files[0]), "pt") as sf:
            meta = sf.metadata() or {}
    else:
        sd = {}
        for f in files:
            with safe_open(str(f), "pt") as sf:
                for k in sf.keys():
                    sd[k] = sf.get_tensor(k)
        save_file(sd, str(d / "weights" / "model.safetensors"), metadata={"format": "pt"})
    tdir = d / "tokenizer"
    if "makeai.tokenizer" in meta:
        from .export import restore_tokenizer_blob
        restore_tokenizer_blob(json.loads(meta["makeai.tokenizer"]), tdir)
    else:
        tdir.mkdir(exist_ok=True)
        for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                  "makeai_tokenizer.json", "spm.model"):
            if (folder / f).exists():
                shutil.copyfile(folder / f, tdir / f)
        if (folder / "generation_config.json").exists():
            shutil.copyfile(folder / "generation_config.json", d / "generation_config.json")


def _import_pt(p: Path, d: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from .export import restore_tokenizer_blob
    obj = torch.load(str(p), map_location="cpu", weights_only=True)
    sd = obj["state_dict"]
    save_file({k: v.contiguous() for k, v in sd.items()}, str(d / "weights" / "model.safetensors"),
              metadata=json.loads(obj.get("makeai_attribution", "{}")) or {"format": "pt"})
    if "makeai_tokenizer" in obj:
        restore_tokenizer_blob(json.loads(obj["makeai_tokenizer"]), d / "tokenizer")


def _onnx_tokenizer(model_path: Path, tdir: Path) -> None:
    import onnx

    from .export import restore_tokenizer_blob
    mp = onnx.load(str(model_path), load_external_data=False)
    meta = {e.key: e.value for e in mp.metadata_props}
    if "makeai.tokenizer" in meta:
        restore_tokenizer_blob(json.loads(meta["makeai.tokenizer"]), tdir)


def _import_package(p: Path, imp: dict, o: dict) -> dict[str, Any]:
    with zipfile.ZipFile(p) as z:
        man = json.loads(z.read("manifest.json"))
        uid = man["uid"]
        d = store.models_dir() / uid
        if (d / "manifest.json").exists():
            raise FileExistsError(f"{man['id']} is already in My AIs")
        names = z.namelist()
        for n in names:
            if ".." in Path(n).parts or Path(n).is_absolute():
                raise ValueError(f"unsafe path in package: {n}")
        base_uid = man.get("base_model")
        install_base = bool(base_uid) and not (store.models_dir() / base_uid / "manifest.json").exists()
        for n in names:
            if n.startswith("base/"):
                if not install_base:
                    continue
                target = store.models_dir() / base_uid / n[len("base/"):]
            elif n == "makeai-package.json":
                continue
            else:
                target = d / n
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    if base_uid and not (store.models_dir() / base_uid / "manifest.json").exists():
        shutil.rmtree(d, ignore_errors=True)
        raise ValueError(f"package needs base model {base_uid}, which is neither installed nor included")
    man["original_creator"] = man.get("original_creator") or man["creator"]
    man["imported_by"] = list(man.get("imported_by") or []) + [imp]
    man["status"] = man.get("status", "imported")
    man["sharing"] = {"visibility": "private", "token": None}
    man["downloads"] = 0
    man.setdefault("provenance", []).append({"event": "imported", "by": imp["username"], "at": imp["at"],
                                             "from": "makeai-package", "path": str(p)})
    store.write_json(d / "manifest.json", man)
    return man
