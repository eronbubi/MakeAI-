"""MakeAI HTTP/WebSocket server. Serves the UI and the local API.

Security model: the API is for the local user. Requests from other machines
(only possible when the user binds to a LAN address for sharing) may reach the
public share pages and package downloads only. State-changing requests must
carry the ``X-MakeAI-Client`` header, which a foreign web page cannot add
without a CORS preflight that this server never approves.
"""
from __future__ import annotations

import asyncio
import html
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .. import PRODUCT, VENDOR, __version__, evaluate, jobs as jobs_mod, projects, registry, sharing, store
from ..data import datasets as D
from ..data.tokenizer import KINDS as TOKENIZER_KINDS, load_tokenizer, train_tokenizer
from ..hardware import benchmark as bench_mod
from ..hardware import recommend as rec_mod
from ..hardware.scanner import Telemetry, scan
from ..inference import InferenceManager
from ..model.config import ModelConfig
from ..train import health
from ..train.checkpoint import CheckpointManager
from ..train.optim import OPTIMIZERS, SCHEDULERS
from ..train.runs import ACTIVE, RunManager, list_runs, read_config, read_metrics, read_status, run_dir, tail_log

try:  # optional development agent - the runtime works without it
    from .. import devagent
except Exception:  # pragma: no cover
    devagent = None

try:  # optional Claude Mode link - the runtime works without it
    from .. import claudemode
except Exception:  # pragma: no cover
    claudemode = None

WEB_DIST = Path(__file__).resolve().parents[1] / "web"
PUBLIC_PREFIXES = ("/s/", "/ai/", "/api/public/", "/favicon", "/assets/share")
MUTATING = ("POST", "PUT", "PATCH", "DELETE")


class State:
    def __init__(self):
        self.telemetry = Telemetry(1.0)
        self.telemetry.start()
        self.runs = RunManager(self.telemetry)
        self.inference = InferenceManager()
        self.jobs = jobs_mod.Jobs()
        self.hw: dict[str, Any] | None = None
        self.hw_at = 0.0
        self.last_auto_opt: dict[str, float] = {}
        self.ui_seen = 0.0             # last time an open MakeAI window received live data (heartbeat)

    def hardware(self, refresh: bool = False) -> dict[str, Any]:
        if self.hw is None or refresh or time.time() - self.hw_at > 300:
            self.hw = scan()
            self.hw_at = time.time()
        return self.hw


def create_app() -> FastAPI:
    app = FastAPI(title=f"{PRODUCT} by {VENDOR}", version=__version__, docs_url="/api/docs", redoc_url=None)
    S = State()
    app.state.S = S

    # ------------------------------------------------------------ security
    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        client = request.client.host if request.client else ""
        local = client in ("127.0.0.1", "::1", "localhost", "testclient")
        if not local and not path.startswith(PUBLIC_PREFIXES):
            return JSONResponse({"detail": "only public share pages are reachable from other computers"}, 403)
        if request.method in MUTATING and not path.startswith("/api/public/") and \
                request.headers.get("x-makeai-client") != "1":
            return JSONResponse({"detail": "missing X-MakeAI-Client header"}, 403)
        return await call_next(request)

    def err(e: Exception, code: int = 400):
        raise HTTPException(code, str(e) or type(e).__name__)

    # ------------------------------------------------------------- basics
    @app.get("/api/health")
    def api_health():
        return {"product": PRODUCT, "vendor": VENDOR, "version": __version__, "home": str(store.home()),
                "dev_agent": devagent is not None, "claude_mode": claudemode is not None, "ui_open": time.time() - S.ui_seen < 15}

    @app.get("/api/settings")
    def get_settings():
        return store.load_settings()

    @app.patch("/api/settings")
    def patch_settings(body: dict = Body(...)):
        if "profile" in body and body["profile"].get("username"):
            try:
                body["profile"]["username"] = registry.normalize_username(body["profile"]["username"])
            except ValueError as e:
                err(e)
        allowed = {"profile", "instant_kill", "auto_optimize", "discover_hubs", "share_host", "share_port"}
        return store.save_settings({k: v for k, v in body.items() if k in allowed})

    def profile() -> dict[str, str]:
        p = store.load_settings()["profile"]
        if not p.get("username"):
            raise HTTPException(400, "set your name and username in Settings first")
        return p

    # ------------------------------------------------------------ hardware
    @app.get("/api/hardware")
    def hardware(refresh: bool = False):
        hw = S.hardware(refresh)
        return {**hw, "benchmark": bench_mod.cached_benchmark()}

    @app.get("/api/hardware/live")
    def hardware_live(seconds: float = 120):
        return {"latest": S.telemetry.latest(), "history": S.telemetry.window(seconds)}

    @app.post("/api/hardware/benchmark")
    def run_bench():
        def work(job):
            job.say("measuring matmul throughput and memory bandwidth")
            return bench_mod.cached_benchmark(force=True)
        return S.jobs.start("benchmark", "Hardware benchmark", work).to_dict()

    @app.post("/api/recommend")
    def recommend(body: dict = Body(...)):
        method = body.get("method", "from_scratch")
        base = None
        if body.get("base_model"):
            base = registry.get_config(body["base_model"])
        tokens = body.get("dataset_tokens")
        if tokens is None and body.get("dataset_ids"):
            tokens = 0
            for ds in body["dataset_ids"]:
                st = D.get_meta(ds).get("stats", {})
                tokens += st.get("tokens") or int(st.get("characters", 0) / 3.6)   # chars/token ~3.6 before tokenising
        try:
            return rec_mod.recommend(S.hardware(), int(body.get("complexity", 4)), method, dataset_tokens=tokens or None,
                                     base_config=base, vocab_size=body.get("vocab_size"),
                                     bench=bench_mod.cached_benchmark(), context_length=body.get("context_length"))
        except ValueError as e:
            err(e)

    @app.post("/api/estimate")
    def estimate(body: dict = Body(...)):
        """Memory/param estimate for a CUSTOM configuration."""
        try:
            cfg = ModelConfig.from_dict(body["model"])
            errs = cfg.validate()
            tr = body.get("training", {})
            mem = rec_mod.estimate_memory(cfg, method=body.get("method", "from_scratch"),
                                          precision=tr.get("precision", "bf16"),
                                          micro_batch=int(tr.get("micro_batch_size", 1)),
                                          ctx=int(tr.get("context_length", cfg.context_length)),
                                          optimizer=tr.get("optimizer", "adamw"),
                                          grad_ckpt=bool(tr.get("gradient_checkpointing")),
                                          lora_rank=int((tr.get("lora") or {}).get("rank", 16)),
                                          base_quant=tr.get("base_quant") or "bf16",
                                          adapter_size=int((tr.get("adapter") or {}).get("bottleneck", 64)))
        except (KeyError, TypeError, ValueError) as e:
            err(e)
        hw = S.hardware()
        dev = hw.get("best_device", {})
        g = next((x for x in hw.get("gpus", []) if x.get("index") == dev.get("index")), {})
        free = ((dev.get("vram_mb") or 0) - (g.get("vram_used_mb") or 0)) * 2**20
        return {"errors": errs, "params": cfg.param_count(), "memory": mem, "vram_free_bytes": free,
                "fits": mem["total"] <= free * 0.95 if free else None, "hf_compatible": cfg.hf_compatible(),
                "attention_kind": cfg.attention_kind}

    @app.get("/api/options")
    def options():
        from ..model import config as C
        return {"activations": C.ACTIVATIONS, "norms": C.NORMS, "positional": C.POSITIONAL,
                "rope_scaling": C.ROPE_SCALING, "attention_types": C.ATTENTION_TYPES, "optimizers": OPTIMIZERS,
                "schedulers": SCHEDULERS, "precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4", "nf4", "fp4"],
                "methods": ["from_scratch", "full", "lora", "qlora", "adapter"], "tokenizers": TOKENIZER_KINDS,
                "complexity": {str(k): v for k, v in rec_mod.COMPLEXITY.items()},
                "lora_ranks": [1, 2, 4, 8, 16, 32, 64, 128], "visibility": sharing.VISIBILITY,
                "discover_categories": sharing.CATEGORIES}

    # ------------------------------------------------------------ files
    @app.post("/api/fs/list")
    def fs_list(body: dict = Body(...)):
        p = Path(body.get("path") or str(Path.home())).expanduser()
        if not p.exists():
            err(FileNotFoundError(str(p)), 404)
        if p.is_file():
            p = p.parent
        items = []
        try:
            for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
                if c.name.startswith("."):
                    continue
                try:
                    items.append({"name": c.name, "path": str(c), "dir": c.is_dir(),
                                  "size": c.stat().st_size if c.is_file() else None})
                except OSError:
                    continue
        except PermissionError as e:
            err(e, 403)
        roots = []
        if os.name == "nt":
            import string
            roots = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
        return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "items": items[:2000],
                "roots": roots, "home": str(Path.home())}

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)):
        d = store.sub("uploads") / uuid.uuid4().hex[:10]
        d.mkdir(parents=True)
        out = []
        for f in files:
            name = Path(f.filename or "file").name
            target = d / name
            with open(target, "wb") as fh:
                while chunk := await f.read(1 << 20):
                    fh.write(chunk)
            out.append(str(target))
        return {"paths": out, "dir": str(d)}

    # ------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    def jobs_list():
        return S.jobs.list()

    @app.get("/api/jobs/{job_id}")
    def job_get(job_id: str):
        try:
            return S.jobs.get(job_id).to_dict(full=True)
        except KeyError:
            err(KeyError("no such job"), 404)

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        S.jobs.get(job_id).cancel.set()
        return {"ok": True}

    # ------------------------------------------------------------ datasets
    @app.get("/api/datasets")
    def ds_list():
        return D.list_datasets()

    @app.post("/api/datasets")
    def ds_create(body: dict = Body(...)):
        name, paths = body.get("name") or "dataset", body.get("paths") or []
        if not paths:
            err(ValueError("choose at least one file or folder"))

        def work(job):
            job.say(f"importing {len(paths)} source(s)")
            return D.create_dataset(name, paths, body.get("options") or {}, progress=lambda n: job.say(f"{n:,} records"))
        return S.jobs.start("dataset-import", f"Import dataset {name}", work).to_dict()

    @app.get("/api/datasets/{ds_id}")
    def ds_get(ds_id: str):
        try:
            return D.get_meta(ds_id)
        except FileNotFoundError as e:
            err(e, 404)

    @app.delete("/api/datasets/{ds_id}")
    def ds_delete(ds_id: str):
        D.delete_dataset(ds_id)
        return {"ok": True}

    @app.get("/api/datasets/{ds_id}/preview")
    def ds_preview(ds_id: str, offset: int = 0, limit: int = 20):
        return D.preview(ds_id, offset, min(limit, 200))

    @app.post("/api/datasets/{ds_id}/{op}")
    def ds_op(ds_id: str, op: str, body: dict = Body(default={})):
        fns = {"clean": lambda: D.clean(ds_id, body), "filter": lambda: D.filter_records(ds_id, body),
               "dedup": lambda: D.dedup(ds_id, body), "shuffle": lambda: D.shuffle(ds_id, int(body.get("seed", 42))),
               "stats": lambda: _stats(ds_id, body)}
        if op not in fns:
            err(ValueError(f"unknown operation {op}"), 404)

        def work(job):
            job.say(f"{op} running")
            return fns[op]()
        return S.jobs.start(f"dataset-{op}", f"{op.title()} {D.get_meta(ds_id)['name']}", work).to_dict()

    def _stats(ds_id: str, body: dict):
        tok = load_tokenizer(_tokenizer_path(body["tokenizer"])) if body.get("tokenizer") else None
        s = D.compute_stats(ds_id, tok)
        meta = D.get_meta(ds_id)
        meta["stats"] = s
        meta.setdefault("token_stats", {})[body.get("tokenizer") or "none"] = s
        store.write_json(store.datasets_dir() / ds_id / "meta.json", meta)
        return s

    # ------------------------------------------------------------ tokenizers
    def _tokenizer_path(ref: str) -> Path:
        """'tok-...' = tokenizer library, 'model:<uid>' = tokenizer of an AI."""
        if ref.startswith("model:"):
            return registry.tokenizer_dir(ref[6:])
        p = store.tokenizers_dir() / ref
        if not p.exists():
            raise HTTPException(404, f"tokenizer {ref} not found")
        return p

    @app.get("/api/tokenizers")
    def tok_list():
        out = []
        for d in sorted(store.tokenizers_dir().iterdir()):
            info = store.read_json(d / "info.json")
            if info:
                out.append(info)
        return out

    @app.post("/api/tokenizers/train")
    def tok_train(body: dict = Body(...)):
        kind = body.get("kind", "bpe")
        name = body.get("name") or f"{kind}-{body.get('vocab_size', 32000)}"
        ds_ids = body.get("dataset_ids") or []
        if not ds_ids:
            err(ValueError("choose at least one dataset to train the tokenizer on"))
        tid = store.new_id("tok")

        def work(job):
            job.say(f"training {kind} tokenizer (vocab {body.get('vocab_size', 32000)})")
            specials = body.get("specials") or None
            extra = body.get("extra_specials")
            tok = train_tokenizer(kind, D.iter_texts(ds_ids, int(body.get("max_chars") or 0)),
                                  int(body.get("vocab_size", 32000)), store.tokenizers_dir() / tid,
                                  specials=specials, extra_specials=extra,
                                  min_frequency=int(body.get("min_frequency", 2)),
                                  spm_model_type=body.get("spm_model_type", "bpe"))
            info = {"id": tid, "name": name, "kind": kind, "vocab_size": tok.vocab_size, "datasets": ds_ids,
                    "created": store.now_iso(), "specials": tok.specials, "extra_specials": tok.extra_specials,
                    "lossless": kind != "wordpiece"}
            store.write_json(store.tokenizers_dir() / tid / "info.json", info)
            return info
        return S.jobs.start("tokenizer", f"Train tokenizer {name}", work).to_dict()

    @app.post("/api/tokenizers/import")
    def tok_import(body: dict = Body(...)):
        src = Path(body["path"])
        src_dir = src if src.is_dir() else src.parent
        tid = store.new_id("tok")
        dst = store.tokenizers_dir() / tid
        dst.mkdir(parents=True)
        for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "makeai_tokenizer.json", "spm.model",
                  "special_tokens_map.json"):
            if (src_dir / f).exists():
                shutil.copyfile(src_dir / f, dst / f)
        try:
            tok = load_tokenizer(dst)
        except Exception as e:
            shutil.rmtree(dst)
            err(e)
        info = {"id": tid, "name": body.get("name") or src_dir.name, "kind": tok.kind, "vocab_size": tok.vocab_size,
                "created": store.now_iso(), "specials": tok.specials, "source": str(src), "lossless": True}
        store.write_json(dst / "info.json", info)
        return info

    @app.post("/api/tokenizers/{ref:path}/encode")
    def tok_encode(ref: str, body: dict = Body(...)):
        tok = load_tokenizer(_tokenizer_path(ref))
        ids = tok.encode(body.get("text", ""))
        return {"ids": ids, "tokens": [tok.id_to_token(i) for i in ids], "count": len(ids),
                "roundtrip": tok.decode(ids, skip_special=False), "describe": tok.describe()}

    @app.delete("/api/tokenizers/{tid}")
    def tok_delete(tid: str):
        shutil.rmtree(_tokenizer_path(tid))
        return {"ok": True}

    # ------------------------------------------------------------ models
    @app.get("/api/models")
    def models_list():
        return registry.list_models()

    @app.get("/api/models/{uid}")
    def model_get(uid: str):
        try:
            m = registry.load_manifest(uid)
        except FileNotFoundError as e:
            err(e, 404)
        cfg = store.read_json(registry.model_dir(uid) / "config.json")
        return {**m, "config": cfg}

    @app.post("/api/models")
    def model_create(body: dict = Body(...)):
        method = body.get("method", "from_scratch")
        try:
            if method == "from_scratch":
                cfg = ModelConfig.from_dict(body["config"])
                tok_dir = _tokenizer_path(body["tokenizer"])
                tok = load_tokenizer(tok_dir)
                if cfg.vocab_size < tok.vocab_size:
                    cfg.vocab_size = tok.vocab_size
                for role, attr in (("bos", "bos_token_id"), ("eos", "eos_token_id"), ("pad", "pad_token_id")):
                    i = tok.special_id(role)
                    if i is not None:
                        setattr(cfg, attr, i)
                errs = cfg.validate()
                if errs:
                    raise ValueError("; ".join(errs))
                base = None
            else:
                base = body["base_model"]
                cfg = registry.get_config(base)
                tok_dir = registry.tokenizer_dir(base)
            m = registry.create_model(name=body["name"], creator_name=body["creator_name"],
                                      creator_username=body["creator_username"], description=body.get("description", ""),
                                      version=str(body.get("version") or "1.0"), tags=body.get("tags") or [],
                                      icon=body.get("icon", ""), config=cfg, method=method, base_model=base,
                                      license=body.get("license", ""))
            shutil.copytree(tok_dir, registry.tokenizer_dir(m["uid"]))
            if body.get("plan"):
                store.write_json(registry.model_dir(m["uid"]) / "training_plan.json", body["plan"])
            return m
        except KeyError as e:
            err(ValueError(f"missing field {e}"))
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            err(e)

    @app.patch("/api/models/{uid}")
    def model_patch(uid: str, body: dict = Body(...)):
        return registry.update_model(uid, body)

    @app.post("/api/models/{uid}/version")
    def model_version(uid: str, body: dict = Body(...)):
        try:
            return registry.new_version(uid, str(body["version"]), profile()["username"])
        except (FileExistsError, KeyError) as e:
            err(e)

    @app.delete("/api/models/{uid}")
    def model_delete(uid: str):
        if S.inference.uid == uid:
            S.inference.unload()
        for r in list_runs():
            if r["model_uid"] == uid and r.get("state") in ACTIVE:
                err(RuntimeError("stop the active training run of this AI first"))
        registry.delete_model(uid)
        return {"ok": True}

    @app.get("/api/models/{uid}/plan")
    def model_plan(uid: str):
        return store.read_json(registry.model_dir(uid) / "training_plan.json", {})

    # ------------------------------------------------------------ import / export
    @app.post("/api/import/inspect")
    def import_inspect(body: dict = Body(...)):
        from ..io.importer import inspect
        try:
            return inspect(body["path"])
        except Exception as e:
            err(e)

    @app.post("/api/import")
    def import_run(body: dict = Body(...)):
        from ..io.importer import import_model
        me = profile()

        def work(job):
            job.say("importing")
            return import_model(body["path"], importer=me, overrides=body.get("overrides") or {})
        return S.jobs.start("import", f"Import {Path(body['path']).name}", work).to_dict()

    @app.get("/api/models/{uid}/export")
    def export_avail(uid: str):
        from ..io.export import availability
        return availability(uid)

    @app.post("/api/models/{uid}/export")
    def export_run(uid: str, body: dict = Body(...)):
        from ..io.export import export
        fmt = body["format"]

        def work(job):
            job.say(f"exporting {fmt}")
            p = export(uid, fmt, body.get("options") or {})
            return {"file": p.name, "bytes": p.stat().st_size, "path": str(p),
                    "download": f"/api/exports/{uid}/{p.name}"}
        return S.jobs.start("export", f"Export {uid} as {fmt}", work).to_dict()

    @app.get("/api/exports/{uid}/{name}")
    def export_download(uid: str, name: str):
        p = (store.exports_dir() / uid / name).resolve()
        if store.exports_dir().resolve() not in p.parents or not p.is_file():
            err(FileNotFoundError(name), 404)
        return FileResponse(p, filename=name)

    # ------------------------------------------------------------ training
    @app.get("/api/runs")
    def runs_list():
        return list_runs()

    @app.post("/api/runs")
    def run_create(body: dict = Body(...)):
        uid = body["model_uid"]
        m = registry.load_manifest(uid)
        for r in list_runs():
            if r.get("state") in ACTIVE and S.runs._alive(r["run_id"]):
                err(RuntimeError(f"run {r['run_id']} is still active - one training run at a time"))
        method = body.get("method") or m["method"]
        if method in ("imported",):
            method = "full"
        cfg: dict[str, Any] = {
            "model_uid": uid, "model_name": m["name"], "method": method,
            "tokenizer_dir": str(registry.tokenizer_dir(uid)),
            "datasets": [{"id": d["id"], "weight": float(d.get("weight", 1))} for d in body["datasets"]],
            "training": body["training"], "checkpoint": body.get("checkpoint") or {}, "eval": body.get("eval") or {},
            "data_prep": body.get("data_prep") or {}, "device": body.get("device"),
        }
        if not cfg["datasets"]:
            err(ValueError("choose at least one dataset"))
        if method == "from_scratch":
            if registry.weights_path(uid).exists() and not body.get("overwrite"):
                cfg["method"] = "full"
                cfg["base_model"] = uid
            else:
                cfg["model_config"] = registry.get_config(uid).to_dict()
        else:
            cfg["base_model"] = m.get("base_model") or uid
        run_id = S.runs.create(cfg)
        m["status"] = "training"
        m.setdefault("training_runs", []).append(run_id)
        registry.save_manifest(m)
        S.runs.start(run_id)
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}")
    def run_get(run_id: str):
        try:
            return {"config": read_config(run_id), "status": read_status(run_id), "alive": S.runs._alive(run_id),
                    "baseline_gpu_util": S.runs.baseline_util.get(run_id)}
        except FileNotFoundError as e:
            err(e, 404)

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str, since: int = 0):
        recs, n = read_metrics(run_id, since)
        return {"records": recs, "next": n}

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: str, lines: int = 300):
        return {"lines": tail_log(run_id, lines)}

    @app.post("/api/runs/{run_id}/{action}")
    def run_action(run_id: str, action: str, body: dict = Body(default={})):
        try:
            if action == "pause":
                return {"seq": S.runs.pause(run_id)}
            if action == "resume":
                return {"seq": S.runs.resume(run_id)}
            if action == "checkpoint":
                return {"seq": S.runs.checkpoint(run_id)}
            if action == "kill":
                instant = bool(body.get("instant")) or bool(body.get("force"))
                return S.runs.kill(run_id, instant=instant)
            if action == "set":
                return {"seq": S.runs.send(run_id, "set", body)}
            if action == "restart":
                ck = body.get("checkpoint") or CheckpointManager(run_dir(run_id) / "checkpoints").latest()
                name = ck["name"] if isinstance(ck, dict) else ck
                if not name:
                    raise RuntimeError("no checkpoint to resume from")
                m = registry.load_manifest(read_config(run_id)["model_uid"])
                m["status"] = "training"
                registry.save_manifest(m)
                return S.runs.start(run_id, resume_from=name)
        except (RuntimeError, FileNotFoundError) as e:
            err(e)
        err(ValueError(f"unknown action {action}"), 404)

    @app.get("/api/runs/{run_id}/checkpoints")
    def run_ckpts(run_id: str):
        cm = CheckpointManager(run_dir(run_id) / "checkpoints")
        return {"checkpoints": cm.index(), "best": cm.best(), "latest": cm.latest()}

    @app.post("/api/runs/{run_id}/checkpoints/{name}/backup")
    def run_ckpt_backup(run_id: str, name: str):
        return {"path": str(CheckpointManager(run_dir(run_id) / "checkpoints").backup(name))}

    def run_health(run_id: str) -> dict[str, Any]:
        cfg = read_config(run_id)
        recs, _ = read_metrics(run_id, 0, ("step",))
        steps = recs[-20:]
        if not steps:
            return {"available": False, "reason": "waiting for the first training steps"}
        # only telemetry from the same time window as these steps (a finished run keeps its final score)
        tel = [s for s in S.telemetry.since(steps[0]["t"] - 1.5) if s["t"] <= steps[-1]["t"] + 1.5]
        hw = S.hardware()
        gpu_static = (hw.get("gpus") or [{}])[0]
        mcfg = None
        try:
            mcfg = registry.get_config(cfg["model_uid"])
        except Exception:
            pass
        fpt = None
        if mcfg:
            fpt = mcfg.flops_per_token() * (1 if cfg["method"] in ("from_scratch", "full") else 2 / 3)
            if cfg["training"].get("gradient_checkpointing"):
                fpt *= 4 / 3
        b = bench_mod.cached_benchmark()
        peak = None
        if b and b.get("tflops"):
            peak = (b["tflops"].get(cfg["training"].get("precision")) or max(b["tflops"].values())) * 1e12
        return health.compute(steps, tel, gpu_static=gpu_static, run_cfg=cfg, flops_per_token=fpt, peak_flops=peak,
                              baseline_gpu_util=S.runs.baseline_util.get(run_id))

    @app.get("/api/runs/{run_id}/health")
    def run_health_api(run_id: str):
        return run_health(run_id)

    # ------------------------------------------------------------ playground
    @app.post("/api/playground/load")
    def pg_load(body: dict = Body(...)):
        try:
            return S.inference.load(body["uid"], body.get("device"), body.get("quant") or None, body.get("n_ctx"))
        except Exception as e:
            err(e)

    @app.post("/api/playground/unload")
    def pg_unload():
        S.inference.unload()
        return {"ok": True}

    @app.get("/api/playground/status")
    def pg_status():
        return S.inference.status()

    @app.post("/api/playground/stop")
    def pg_stop():
        S.inference.stop()
        return {"ok": True}

    @app.post("/api/playground/chat")
    def pg_chat(body: dict = Body(...)):
        def gen():
            try:
                for chunk in S.inference.stream(body["uid"], body.get("messages") or [], body.get("params") or {},
                                                system_prompt=body.get("system_prompt") or None,
                                                raw_prompt=body.get("raw_prompt"), quant=body.get("quant") or None):
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e) or type(e).__name__})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------ evaluation
    @app.post("/api/eval/dataset")
    def eval_dataset(body: dict = Body(...)):
        def work(job):
            if S.inference.uid:
                S.inference.unload()
            r = evaluate.dataset_metrics(body["uid"], body["dataset_id"], body.get("split", "val"),
                                         int(body.get("max_batches", 50)), progress=job.say)
            return evaluate.record(body["uid"], r, body.get("name") or f"dataset {body['dataset_id']}")
        return S.jobs.start("eval", "Evaluate on dataset", work).to_dict()

    @app.post("/api/eval/benchmark")
    def eval_bench(body: dict = Body(...)):
        items = body.get("items") or (evaluate.load_benchmark_file(body["path"]) if body.get("path") else None)
        if not items:
            err(ValueError("provide benchmark items or a .json/.jsonl file"))

        def work(job):
            r = evaluate.run_benchmark(body["uid"], items, S.inference, body.get("params"), progress=job.say)
            entry = evaluate.record(body["uid"], r, body.get("name") or "benchmark")
            return {**entry, "results": r["results"]}
        return S.jobs.start("eval", f"Benchmark ({len(items)} items)", work).to_dict()

    @app.get("/api/eval/compare")
    def eval_compare(uids: str):
        return evaluate.compare([u for u in uids.split(",") if u])

    # ------------------------------------------------------------ sharing / discover
    def base_url(request: Request) -> str:
        return str(request.base_url).rstrip("/")

    @app.post("/api/models/{uid}/share")
    def share(uid: str, request: Request, body: dict = Body(...)):
        try:
            m = sharing.set_visibility(uid, body["visibility"])
        except ValueError as e:
            err(e)
        return sharing.public_card(m, base_url(request)) | {"visibility": m["sharing"]["visibility"]}

    @app.get("/api/public/index")
    def public_index(request: Request):
        return {"instance": f"{PRODUCT} {__version__}", "items": sharing.public_index(base_url(request))}

    @app.get("/api/public/package/{token}")
    def public_package(token: str):
        m = sharing.by_token(token)
        if not m:
            err(FileNotFoundError("not shared"), 404)
        from ..io.export import export_package
        p = export_package(m["uid"])
        sharing.count_download(m["uid"])
        return FileResponse(p, filename=p.name, media_type="application/zip")

    @app.get("/api/discover")
    def discover(request: Request, category: str | None = None, q: str = ""):
        return sharing.discover(category, q, base_url(request))

    @app.post("/api/discover/install")
    def discover_install(body: dict = Body(...)):
        from ..io.importer import import_model
        me = profile()

        def work(job):
            job.say("downloading package")
            path = sharing.download_to_temp(body["download"])
            job.say("importing")
            try:
                return import_model(path, importer=me)
            finally:
                os.unlink(path)
        return S.jobs.start("install", f"Install {body.get('name', 'AI')}", work).to_dict()

    @app.get("/s/{token}", response_class=HTMLResponse)
    def share_page(token: str, request: Request):
        m = sharing.by_token(token)
        if not m:
            return HTMLResponse(_page("Not found", "<p>This AI is not shared (or the link was revoked).</p>"), 404)
        return HTMLResponse(_ai_page(sharing.public_card(m, base_url(request))))

    @app.get("/ai/{username}/{slug}/{version}", response_class=HTMLResponse)
    def public_page(username: str, slug: str, version: str, request: Request):
        m = sharing.by_path(username, slug, version)
        if not m:
            return HTMLResponse(_page("Not found", "<p>No public AI at this address.</p>"), 404)
        return HTMLResponse(_ai_page(sharing.public_card(m, base_url(request))))

    # ------------------------------------------------------------ projects
    @app.get("/api/projects")
    def proj_list():
        return projects.list_projects()

    @app.post("/api/projects")
    def proj_create(body: dict = Body(...)):
        try:
            return projects.create(body["name"], body.get("description", ""))
        except FileExistsError as e:
            err(e)

    @app.get("/api/projects/{name}")
    def proj_get(name: str):
        return {"project": projects.get(name), "tree": projects.tree(name)}

    @app.get("/api/projects/{name}/file")
    def proj_read(name: str, path: str):
        try:
            return projects.read_file(name, path)
        except (PermissionError, FileNotFoundError) as e:
            err(e, 404)

    @app.put("/api/projects/{name}/file")
    def proj_write(name: str, body: dict = Body(...)):
        try:
            return projects.write_file(name, body["path"], body.get("content", ""))
        except (PermissionError, IsADirectoryError) as e:
            err(e)

    @app.post("/api/projects/{name}/mkdir")
    def proj_mkdir(name: str, body: dict = Body(...)):
        projects.make_dir(name, body["path"])
        return {"ok": True}

    @app.delete("/api/projects/{name}/file")
    def proj_delete(name: str, path: str):
        try:
            projects.delete_path(name, path)
        except PermissionError as e:
            err(e)
        return {"ok": True}

    @app.post("/api/projects/{name}/link")
    def proj_link(name: str, body: dict = Body(...)):
        try:
            return projects.link(name, body["kind"], body["ref"], bool(body.get("on", True)))
        except ValueError as e:
            err(e)

    # ------------------------------------------------------------ requirements / tests
    @app.get("/api/requirements")
    def requirements():
        root = Path(__file__).resolve().parents[2]
        req = store.read_json(root / "requirements.json", {})
        res = store.read_json(store.home() / "test_results.json", {})
        return {"requirements": req, "results": res}

    @app.post("/api/tests/run")
    def tests_run(body: dict = Body(default={})):
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[2]

        def work(job):
            xml = store.home() / "test_results.xml"
            args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"]
            if body.get("quick"):
                args += ["-m", "not slow"]
            job.say("running " + " ".join(args[2:]))
            env = dict(os.environ)
            env.pop("MAKEAI_HOME", None)   # tests use their own temporary home
            p = subprocess.Popen(args, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", env=env,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in p.stdout:
                job.say(line.rstrip())
            code = p.wait()
            res = _parse_junit(xml)
            res["exit_code"] = code
            res["at"] = store.now_iso()
            store.write_json(store.home() / "test_results.json", res)
            return res
        return S.jobs.start("tests", "Acceptance tests", work).to_dict()

    # ------------------------------------------------------------ dev agent (optional)
    @app.get("/api/dev/status")
    def dev_status():
        if devagent is None:
            return {"available": False, "installed": False,
                    "reason": "the optional development agent module is not installed"}
        return {"installed": True, **devagent.status()}

    @app.post("/api/dev/plan")
    def dev_plan(body: dict = Body(...)):
        if devagent is None:
            err(RuntimeError("development agent not installed"), 404)

        def work(job):
            job.say("Analyzing project...")
            r = devagent.plan(body["task"], cancel=job.cancel)
            job.say("plan ready")
            return r
        return S.jobs.start("dev-plan", "Plan: " + body["task"][:60], work).to_dict()

    @app.post("/api/dev/execute")
    def dev_exec(body: dict = Body(...)):
        if devagent is None:
            err(RuntimeError("development agent not installed"), 404)

        def work(job):
            job.say("implementing approved plan")
            return devagent.execute(body["plan"], body.get("session_id"),
                                    on_line=lambda l: l and job.say(l), cancel=job.cancel)
        return S.jobs.start("dev-exec", "Implement plan", work).to_dict()

    # ------------------------------------------------------------ live websocket
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        origin = ws.headers.get("origin") or ""
        host = ws.headers.get("host") or ""
        client = ws.client.host if ws.client else ""
        if client not in ("127.0.0.1", "::1", "testclient") or (origin and host not in origin):
            await ws.close(code=4403)
            return
        await ws.accept()
        subscribed: str | None = None
        offset = 0
        last_health = 0.0
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                    S.ui_seen = time.time()      # the window is alive only if it talks to us (pings every 5 s)
                    if "subscribe" in msg:
                        subscribed = msg["subscribe"]
                        offset = int(msg.get("since", 0))
                except asyncio.TimeoutError:
                    pass
                payload: dict[str, Any] = {"type": "live", "t": time.time(), "telemetry": S.telemetry.latest()}
                active = []
                for r in list_runs()[:20]:
                    if r.get("state") in ACTIVE:
                        active.append(r)
                payload["active_runs"] = active
                if subscribed:
                    try:
                        recs, offset2 = await loop.run_in_executor(None, read_metrics, subscribed, offset)
                        payload["run"] = {"id": subscribed, "status": read_status(subscribed), "records": recs,
                                          "next": offset2, "alive": S.runs._alive(subscribed)}
                        offset = offset2
                        if time.time() - last_health >= 2:
                            h = await loop.run_in_executor(None, run_health, subscribed)
                            payload["run"]["health"] = h
                            last_health = time.time()
                            _auto_optimize(subscribed, h)
                    except FileNotFoundError:
                        payload["run"] = None
                await ws.send_json(_json_safe(payload))
        except (WebSocketDisconnect, RuntimeError):
            return

    def _auto_optimize(run_id: str, h: dict) -> None:
        """Apply live-adjustable recommendations only when the user enabled Auto Optimization."""
        if not h.get("available") or not store.load_settings().get("auto_optimize"):
            return
        if time.time() - S.last_auto_opt.get(run_id, 0) < 60:
            return
        if (h.get("detail", {}).get("data_wait_pct") or 0) > 8 and read_status(run_id).get("state") == "running":
            cur = int(read_config(run_id)["training"].get("dataloader_workers", 2))
            new = min(32, cur * 2)
            if new != cur:
                S.runs.send(run_id, "set", {"dataloader_workers": new})
                cfg = read_config(run_id)
                cfg["training"]["dataloader_workers"] = new
                store.write_json(run_dir(run_id) / "run.json", cfg)
                S.last_auto_opt[run_id] = time.time()

    # ------------------------------------------------------------ Claude Mode (optional)
    if claudemode is not None:
        claudemode.install(app, S, list_runs, read_status)

    # ------------------------------------------------------------ UI
    @app.get("/favicon.svg")
    def favicon():
        return FileResponse(WEB_DIST / "favicon.svg") if (WEB_DIST / "favicon.svg").exists() else HTMLResponse("", 404)

    if (WEB_DIST / "assets").exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        index = WEB_DIST / "index.html"
        if not index.exists():
            return HTMLResponse(_page("MakeAI", "<p>The UI has not been built. Run <code>npm run build</code> in "
                                               "<code>ui/</code>. The API is available at <a href='/api/docs'>/api/docs</a>.</p>"))
        # never cache the entry page: after an update it must point at the new hashed chunks
        return FileResponse(index, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    @app.on_event("shutdown")
    def _shutdown():
        S.telemetry.stop()
        S.runs.shutdown()
        S.inference.unload()

    return app


def _json_safe(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


def _parse_junit(path: Path) -> dict[str, Any]:
    import xml.etree.ElementTree as ET
    if not path.exists():
        return {"tests": []}
    root = ET.parse(path).getroot()
    tests = []
    for tc in root.iter("testcase"):
        state = "passed"
        msg = ""
        for tag in ("failure", "error", "skipped"):
            el = tc.find(tag)
            if el is not None:
                state = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                msg = (el.get("message") or "")[:500]
        tests.append({"name": tc.get("name"), "classname": tc.get("classname"), "time": float(tc.get("time") or 0),
                      "state": state, "message": msg})
    return {"tests": tests, "passed": sum(t["state"] == "passed" for t in tests),
            "failed": sum(t["state"] in ("failed", "error") for t in tests),
            "skipped": sum(t["state"] == "skipped" for t in tests)}


# ---------------------------------------------------------------- share pages
def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · MakeAI</title><link rel="icon" href="/favicon.svg">
<style>
:root{{--bg:#ffffff;--panel:#ffffff;--line:#e8e3f1;--text:#121016;--dim:#5c5569;--acc:#7c3aed;--ink:#0d0b12}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;border-top:6px solid var(--acc)}}
footer{{background:var(--ink);color:#a9a2b8;padding:10px 14px;border-radius:6px}}
main{{max-width:760px;margin:0 auto;padding:40px 16px}} h1{{font-size:26px;margin:0 0 4px}} .dim{{color:var(--dim)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px;margin-top:20px}}
dl{{display:grid;grid-template-columns:170px 1fr;gap:8px 16px;margin:0}} dt{{color:var(--dim)}} dd{{margin:0;overflow-wrap:anywhere}}
.tag{{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 6px;margin:0 4px 4px 0;font-size:12px}}
a.btn{{display:inline-block;margin-top:18px;background:var(--acc);color:#fff;padding:8px 14px;border-radius:6px;text-decoration:none;font-weight:600}}
code{{font-family:ui-monospace,Consolas,monospace;font-size:12.5px}} footer{{margin-top:28px;font-size:12px}}
@media (max-width:560px){{dl{{grid-template-columns:1fr}} dt{{margin-top:6px}}}}
</style></head><body><main>{body}<footer class="dim">Shared with MakeAI by Convergent</footer></main></body></html>"""


def _fmt_params(n):
    if not n:
        return "unknown"
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def _ai_page(c: dict[str, Any]) -> str:
    e = html.escape
    oc = c.get("original_creator") or c["creator"]
    rh = c.get("required_hardware") or {}
    hw = ", ".join(f"{k.replace('_', ' ')}: {v} GB" for k, v in rh.items()) or "not specified"
    tags = "".join(f"<span class='tag'>{e(t)}</span>" for t in c.get("tags") or []) or "<span class='dim'>none</span>"
    imp = ""
    if c.get("imported_by"):
        imp = f"<dt>Imported by</dt><dd>{', '.join(e(i['name']) + ' (@' + e(i['username']) + ')' for i in c['imported_by'])}</dd>"
    orig = ""
    if oc.get("username") != c["creator"].get("username"):
        orig = f"<dt>Originally created by</dt><dd>{e(oc['name'])} (@{e(oc['username'])})</dd>"
    body = f"""<div class="dim">{e(c['id'])}</div><h1>{e(c.get('icon') or '')} {e(c['name'])}</h1>
<div>Created by <b>{e(c['creator']['name'])}</b> <span class="dim">@{e(c['creator']['username'])}</span></div>
<div class="card"><p style="margin-top:0">{e(c.get('description') or '')}</p><dl>
<dt>Creator</dt><dd>{e(c['creator']['name'])}</dd><dt>Username</dt><dd>@{e(c['creator']['username'])}</dd>{orig}{imp}
<dt>Version</dt><dd>{e(str(c['version']))}</dd><dt>Parameters</dt><dd>{_fmt_params(c.get('param_count'))}</dd>
<dt>Context</dt><dd>{e(str(c.get('context_length') or 'unknown'))} tokens</dd><dt>Tags</dt><dd>{tags}</dd>
<dt>Created</dt><dd>{e(str(c.get('created') or ''))}</dd><dt>Last update</dt><dd>{e(str(c.get('updated') or ''))}</dd>
<dt>Required hardware</dt><dd>{e(hw)}</dd><dt>Model format</dt><dd>{e(str(c.get('format')))} · runs with {e(str(c.get('backend')))}</dd>
<dt>Downloads</dt><dd>{c.get('downloads', 0)}</dd></dl>
{f"<a class='btn' href='{e(c['download'])}'>Download .makeai package</a>" if c.get('download') else ''}</div>"""
    return _page(c["name"], body)
