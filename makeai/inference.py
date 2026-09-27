"""Local inference runtimes for the Playground. No network, no cloud API.

Backends:
    native       MakeAI transformer (models created in MakeAI, Llama/Mistral/Qwen2 safetensors)
    llama.cpp    GGUF files via llama-cpp-python
    onnxruntime  ONNX exports (full recompute per token - no KV cache in the graph)
"""
from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from . import registry
from .model.generate import SamplingParams, build_chat_prompt, generate_stream, process_logits, sample, StreamDecoder


class NativeBackend:
    name = "native"

    def __init__(self, uid: str, device: str, quant: str | None):
        import torch
        self.model, self.tok, self.manifest = registry.load_model(uid, device=device, quant=quant)
        self.device = device
        self.quant = quant
        self.ctx = self.model.cfg.context_length
        self.dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")

    def stream(self, messages, params: SamplingParams, system_prompt: str | None, raw_prompt: str | None,
               cancel: threading.Event) -> Iterator[dict]:
        if raw_prompt is not None:
            ids = self.tok.encode(raw_prompt)
            if self.tok.bos_id is not None and self.tok.kind != "custom":
                ids = [self.tok.bos_id] + ids
            dropped = 0
        else:
            ids, dropped = build_chat_prompt(self.tok, messages, system_prompt, self.ctx, params.max_tokens)
        if dropped:
            yield {"info": f"{dropped} earliest message(s) dropped to fit the {self.ctx}-token context"}
        stops = self.tok.chat_stop_tokens() if raw_prompt is None else [i for i in [self.tok.eos_id] if i is not None]
        yield from generate_stream(self.model, self.tok, ids, params, stop_token_ids=stops,
                                   should_stop=cancel.is_set)

    def close(self):
        del self.model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


class LlamaCppBackend:
    name = "llama.cpp"

    def __init__(self, uid: str, n_ctx: int | None = None):
        import llama_cpp
        path = registry.model_dir(uid) / "weights" / "model.gguf"
        self.manifest = registry.load_manifest(uid)
        ctx = n_ctx or min(int(self.manifest.get("context_length") or 4096), 8192)
        self.llm = llama_cpp.Llama(model_path=str(path), n_ctx=ctx, n_gpu_layers=-1, verbose=False)
        self.ctx = ctx
        self.device = "cpu" if not llama_cpp.llama_supports_gpu_offload() else "gpu"
        self.quant = None
        self.dtype = "gguf"

    def stream(self, messages, params: SamplingParams, system_prompt, raw_prompt, cancel) -> Iterator[dict]:
        kw = dict(temperature=params.temperature, top_k=params.top_k, top_p=params.top_p, min_p=params.min_p,
                  repeat_penalty=params.repetition_penalty, frequency_penalty=params.frequency_penalty,
                  presence_penalty=params.presence_penalty, max_tokens=params.max_tokens, seed=params.seed,
                  stop=params.stop or None, stream=True)
        t0 = time.perf_counter()
        first = None
        n = 0
        text = ""
        finish = "length"
        if raw_prompt is not None:
            it = self.llm.create_completion(raw_prompt, **kw)
            get = lambda ch: ch["choices"][0].get("text", "")
        else:
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                   [m for m in messages if m["role"] != "system" or not system_prompt]
            it = self.llm.create_chat_completion(messages=msgs, **kw)
            get = lambda ch: ch["choices"][0]["delta"].get("content", "") or ""
        for ch in it:
            d = get(ch)
            fr = ch["choices"][0].get("finish_reason")
            if d:
                if first is None:
                    first = time.perf_counter() - t0
                n += 1
                text += d
                yield {"delta": d}
            if fr:
                finish = {"stop": "stop_token"}.get(fr, fr)
            if cancel.is_set():
                finish = "cancelled"
                break
        total = time.perf_counter() - t0
        yield {"done": True, "finish_reason": finish, "completion_tokens": n, "time_to_first_token_s": round(first or 0, 4),
               "total_s": round(total, 3), "tokens_per_s": round((n - 1) / (total - (first or 0)), 1) if n > 1 else None,
               "text": text, "note": "token count = streamed chunks"}

    def close(self):
        self.llm.close() if hasattr(self.llm, "close") else None
        del self.llm
        gc.collect()


class OnnxBackend:
    name = "onnxruntime"

    def __init__(self, uid: str):
        import onnxruntime as ort

        from .data.tokenizer import load_tokenizer
        self.manifest = registry.load_manifest(uid)
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(registry.model_dir(uid) / "weights" / "model.onnx"), providers=providers)
        self.tok = load_tokenizer(registry.tokenizer_dir(uid))
        cfg = registry.get_config(uid)
        self.ctx = cfg.context_length
        self.device = providers[0]
        self.quant = None
        self.dtype = "fp32"

    def stream(self, messages, params: SamplingParams, system_prompt, raw_prompt, cancel) -> Iterator[dict]:
        import numpy as np
        import torch
        if raw_prompt is not None:
            ids = self.tok.encode(raw_prompt)
            if self.tok.bos_id is not None and self.tok.kind != "custom":
                ids = [self.tok.bos_id] + ids
        else:
            ids, _ = build_chat_prompt(self.tok, messages, system_prompt, self.ctx, params.max_tokens)
        stops = set(self.tok.chat_stop_tokens())
        gen = torch.Generator()
        if params.seed is not None:
            gen.manual_seed(int(params.seed))
        dec = StreamDecoder(self.tok)
        counts: dict[int, int] = {}
        t0 = time.perf_counter()
        first = None
        n = 0
        text = ""
        finish = "length"
        for _ in range(params.max_tokens):
            window = ids[-self.ctx:]
            logits = self.sess.run(None, {"input_ids": np.asarray([window], dtype=np.int64)})[0][0, -1]
            lg = process_logits(torch.from_numpy(logits.copy()), ids, counts, params)
            nxt = sample(lg, params, gen if params.seed is not None else None)
            if first is None:
                first = time.perf_counter() - t0
            if nxt in stops:
                finish = "stop_token"
                break
            ids.append(nxt)
            counts[nxt] = counts.get(nxt, 0) + 1
            n += 1
            d = dec.push(nxt)
            if d:
                text += d
                hit = next((s for s in params.stop if s in text), None)
                if hit:
                    finish = "stop_sequence"
                    text = text[:text.find(hit)]
                    break
                yield {"delta": d}
            if cancel.is_set():
                finish = "cancelled"
                break
        total = time.perf_counter() - t0
        yield {"done": True, "finish_reason": finish, "completion_tokens": n, "time_to_first_token_s": round(first or 0, 4),
               "total_s": round(total, 3), "tokens_per_s": round((n - 1) / (total - (first or 0)), 1) if n > 1 else None,
               "text": text}

    def close(self):
        del self.sess
        gc.collect()


class InferenceManager:
    """Keeps at most one model loaded; loading another unloads the previous one first."""

    def __init__(self):
        self.backend = None
        self.uid: str | None = None
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.loaded_at: float | None = None
        self.load_seconds: float | None = None

    def load(self, uid: str, device: str | None = None, quant: str | None = None, n_ctx: int | None = None) -> dict:
        m = registry.load_manifest(uid)
        if not m.get("runnable"):
            raise ValueError(m.get("not_runnable_reason") or f"{m['name']} is not trained yet")
        with self.lock:
            key = (uid, device, quant)
            if self.backend is not None and getattr(self, "_key", None) == key:
                return self.status()
            self.unload()
            t0 = time.time()
            backend = m.get("backend") or "native"
            if backend == "llama.cpp":
                self.backend = LlamaCppBackend(uid, n_ctx)
            elif backend == "onnxruntime":
                self.backend = OnnxBackend(uid)
            else:
                import torch
                dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.backend = NativeBackend(uid, dev, quant)
            self.uid, self._key = uid, key
            self.loaded_at = time.time()
            self.load_seconds = round(time.time() - t0, 2)
            return self.status()

    def unload(self):
        with self.lock:
            if self.backend is not None:
                self.backend.close()
            self.backend, self.uid, self._key = None, None, None

    def status(self) -> dict:
        if self.backend is None:
            return {"loaded": False}
        st = {"loaded": True, "uid": self.uid, "backend": self.backend.name, "device": self.backend.device,
              "quant": self.backend.quant, "dtype": self.backend.dtype, "context_length": self.backend.ctx,
              "load_seconds": self.load_seconds}
        try:
            import torch
            if str(self.backend.device).startswith("cuda"):
                st["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20)
        except Exception:
            pass
        return st

    def stream(self, uid: str, messages: list[dict], params: dict, system_prompt: str | None = None,
               raw_prompt: str | None = None, device: str | None = None, quant: str | None = None) -> Iterator[dict]:
        self.load(uid, device, quant)
        sp = SamplingParams.from_dict(params)
        self.cancel.clear()
        with self.lock:
            yield from self.backend.stream(messages, sp, system_prompt, raw_prompt, self.cancel)

    def stop(self):
        self.cancel.set()
