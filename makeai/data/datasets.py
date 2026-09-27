"""Dataset system: import, preview, cleaning, filtering, dedup, shuffle, split, tokenise, pack.

A dataset is a directory ``datasets/<id>/`` holding ``meta.json`` and
``records.jsonl``. Each record is either ``{"text": ...}`` or
``{"messages": [{"role", "content"}, ...]}``. Every processing step streams the
file, so datasets larger than RAM work.

``prepare()`` tokenises a dataset for one tokenizer and context length into
flat token/loss-mask streams (``<split>.tokens.bin`` / ``<split>.mask.bin``)
that the trainer memory-maps.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .. import store

TEXT_EXTS = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
             ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".ps1", ".sql", ".html", ".css", ".xml", ".yaml", ".yml",
             ".toml", ".ini", ".tex", ".kt", ".swift", ".lua", ".r", ".m", ".scala", ".dart", ".vue", ".svelte"}
STRUCTURED_EXTS = {".json", ".jsonl", ".csv", ".parquet"}
TEXT_FIELDS = ("text", "content", "body", "document", "code", "article", "story")


# ================================================================== records
def _to_record(obj: Any, text_field: str | None = None) -> dict | None:
    """Normalise one source row into a MakeAI record, or None if unusable."""
    if isinstance(obj, str):
        return {"text": obj} if obj.strip() else None
    if not isinstance(obj, dict):
        return None
    if text_field:
        v = obj.get(text_field)
        return {"text": str(v)} if v is not None and str(v).strip() else None
    msgs = obj.get("messages") or obj.get("conversations") or obj.get("conversation")
    if isinstance(msgs, list) and msgs:
        out = []
        role_map = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant", "user": "user",
                    "assistant": "assistant", "system": "system"}
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = role_map.get(str(m.get("role") or m.get("from") or "").lower())
            content = m.get("content") if "content" in m else m.get("value")
            if role and content is not None:
                out.append({"role": role, "content": str(content)})
        return {"messages": out} if any(m["role"] == "assistant" for m in out) else None
    pairs = [("instruction", "output"), ("prompt", "response"), ("prompt", "completion"),
             ("question", "answer"), ("input", "output"), ("query", "response")]
    for p, r in pairs:
        if p in obj and r in obj and obj[r] is not None:
            user = str(obj[p])
            if p == "instruction" and obj.get("input"):
                user = f"{user}\n\n{obj['input']}"
            msgs = []
            if obj.get("system"):
                msgs.append({"role": "system", "content": str(obj["system"])})
            msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": str(obj[r])}]
            return {"messages": msgs}
    for f in TEXT_FIELDS:
        if f in obj and obj[f] is not None and str(obj[f]).strip():
            return {"text": str(obj[f])}
    return None


def iter_source(path: Path, opts: dict[str, Any]) -> Iterator[dict]:
    """Yield records from one file of any supported format."""
    ext = path.suffix.lower()
    tf = opts.get("text_field") or None
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = _to_record(json.loads(line), tf)
                    except json.JSONDecodeError:
                        continue
                    if r:
                        yield r
    elif ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        for obj in data:
            r = _to_record(obj, tf)
            if r:
                yield r
    elif ext == ".csv":
        import pandas as pd
        for chunk in pd.read_csv(path, chunksize=10_000, dtype=str, keep_default_na=False):
            for obj in chunk.to_dict("records"):
                r = _to_record(obj, tf)
                if r:
                    yield r
    elif ext == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096):
            for obj in batch.to_pylist():
                r = _to_record(obj, tf)
                if r:
                    yield r
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        mode = opts.get("txt_mode", "file")
        if mode == "line":
            parts = text.splitlines()
        elif mode == "paragraph":
            parts = re.split(r"\n\s*\n", text)
        else:
            parts = [text]
        for p in parts:
            if p.strip():
                yield {"text": p}


def expand_sources(paths: list[str], opts: dict[str, Any]) -> list[Path]:
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in (opts.get("extensions") or [])}
    allowed = exts or (TEXT_EXTS | STRUCTURED_EXTS)
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
                for fn in sorted(files):
                    fp = Path(root) / fn
                    if fp.suffix.lower() in allowed:
                        out.append(fp)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(str(p))
    return out


# ================================================================== dataset
def ds_dir(ds_id: str) -> Path:
    d = store.datasets_dir() / ds_id
    if not d.exists():
        raise FileNotFoundError(f"dataset {ds_id} not found")
    return d


def list_datasets() -> list[dict]:
    out = []
    for d in sorted(store.datasets_dir().iterdir()):
        m = store.read_json(d / "meta.json")
        if m:
            out.append(m)
    return out


def get_meta(ds_id: str) -> dict:
    return store.read_json(ds_dir(ds_id) / "meta.json")


def _save_meta(meta: dict) -> None:
    meta["updated"] = store.now_iso()
    store.write_json(store.datasets_dir() / meta["id"] / "meta.json", meta)


def create_dataset(name: str, paths: list[str], opts: dict[str, Any] | None = None,
                   progress: Callable[[int], None] | None = None) -> dict:
    opts = opts or {}
    files = expand_sources(paths, opts)
    if not files:
        raise ValueError("no supported files found in the given paths")
    ds_id = store.new_id("ds")
    d = store.datasets_dir() / ds_id
    d.mkdir(parents=True)
    n = 0
    with open(d / "records.jsonl", "w", encoding="utf-8") as out:
        for fp in files:
            for rec in iter_source(fp, opts):
                if opts.get("include_source_path") and "text" in rec:
                    rec["source"] = str(fp)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if progress and n % 1000 == 0:
                    progress(n)
    meta = {"id": ds_id, "name": name, "sources": [str(p) for p in paths], "files": len(files),
            "options": opts, "created": store.now_iso(), "history": [{"op": "import", "files": len(files),
                                                                        "records": n, "at": store.now_iso()}]}
    meta["stats"] = compute_stats(ds_id, meta_override=meta)
    _save_meta(meta)
    return meta


def create_from_records(name: str, records: Iterable[dict], source: str = "upload") -> dict:
    ds_id = store.new_id("ds")
    d = store.datasets_dir() / ds_id
    d.mkdir(parents=True)
    n = 0
    with open(d / "records.jsonl", "w", encoding="utf-8") as out:
        for r in records:
            r = _to_record(r)
            if r:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    meta = {"id": ds_id, "name": name, "sources": [source], "files": 1, "options": {}, "created": store.now_iso(),
            "history": [{"op": "import", "records": n, "at": store.now_iso()}]}
    meta["stats"] = compute_stats(ds_id, meta_override=meta)
    _save_meta(meta)
    return meta


def delete_dataset(ds_id: str) -> None:
    import shutil
    shutil.rmtree(ds_dir(ds_id))


def iter_records(ds_id: str) -> Iterator[dict]:
    with open(ds_dir(ds_id) / "records.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def record_text(r: dict) -> str:
    if "text" in r:
        return r["text"]
    return "\n".join(f"{m['role']}: {m['content']}" for m in r.get("messages", []))


def preview(ds_id: str, offset: int = 0, limit: int = 20) -> list[dict]:
    out = []
    for i, r in enumerate(iter_records(ds_id)):
        if i < offset:
            continue
        if len(out) >= limit:
            break
        out.append({"index": i, **r})
    return out


def compute_stats(ds_id: str, tokenizer=None, meta_override: dict | None = None) -> dict:
    n = chars = words = max_chars = 0
    kinds = {"text": 0, "chat": 0}
    tok_total = tok_max = 0
    batch: list[str] = []

    def flush():
        nonlocal tok_total, tok_max, batch
        if tokenizer is not None and batch:
            for ids in tokenizer.encode_batch(batch):
                tok_total += len(ids)
                tok_max = max(tok_max, len(ids))
        batch = []

    for r in iter_records(ds_id):
        t = record_text(r)
        n += 1
        chars += len(t)
        words += len(t.split())
        max_chars = max(max_chars, len(t))
        kinds["chat" if "messages" in r else "text"] += 1
        if tokenizer is not None:
            batch.append(t)
            if len(batch) >= 512:
                flush()
    flush()
    size = (store.datasets_dir() / ds_id / "records.jsonl").stat().st_size
    s = {"samples": n, "characters": chars, "words": words, "bytes": size,
         "avg_chars": round(chars / n, 1) if n else 0, "max_chars": max_chars, "kinds": kinds}
    if tokenizer is not None:
        s.update({"tokens": tok_total, "avg_tokens": round(tok_total / n, 1) if n else 0, "max_tokens": tok_max,
                  "tokenizer_vocab": tokenizer.vocab_size})
    return s


def _rewrite(ds_id: str, fn: Callable[[Iterator[dict]], Iterator[dict]], op: dict) -> dict:
    d = ds_dir(ds_id)
    src = d / "records.jsonl"
    tmp = d / "records.jsonl.tmp"
    before = kept = 0

    def counted():
        nonlocal before
        for r in iter_records(ds_id):
            before += 1
            yield r

    with open(tmp, "w", encoding="utf-8") as out:
        for r in fn(counted()):
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp, src)
    meta = get_meta(ds_id)
    op.update({"before": before, "after": kept, "removed": before - kept, "at": store.now_iso()})
    meta.setdefault("history", []).append(op)
    meta["stats"] = compute_stats(ds_id)
    _save_meta(meta)
    return op


# ================================================================ cleaning
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(t: str, o: dict) -> str:
    if o.get("unicode_nfc", True):
        t = unicodedata.normalize("NFC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    if o.get("strip_control", True):
        t = _CTRL.sub("", t)
    if o.get("strip_trailing_whitespace", True):
        t = "\n".join(line.rstrip() for line in t.split("\n"))
    if o.get("collapse_blank_lines", True):
        t = re.sub(r"\n{4,}", "\n\n\n", t)
    if o.get("strip_html", False):
        t = re.sub(r"<[^>]+>", "", t)
    return t.strip("\n") if o.get("strip_edges", True) else t


def clean(ds_id: str, opts: dict | None = None) -> dict:
    o = opts or {}

    def fn(records):
        for r in records:
            if "text" in r:
                t = _clean_text(r["text"], o)
                if t.strip():
                    yield {**r, "text": t}
            else:
                msgs = [{**m, "content": _clean_text(m["content"], o)} for m in r["messages"]]
                if any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
                    yield {**r, "messages": msgs}
    return _rewrite(ds_id, fn, {"op": "clean", "options": o})


def filter_records(ds_id: str, opts: dict) -> dict:
    min_c, max_c = int(opts.get("min_chars") or 0), int(opts.get("max_chars") or 0)
    min_w, max_w = int(opts.get("min_words") or 0), int(opts.get("max_words") or 0)
    inc = re.compile(opts["include_regex"]) if opts.get("include_regex") else None
    exc = re.compile(opts["exclude_regex"]) if opts.get("exclude_regex") else None
    max_non_alnum = float(opts.get("max_symbol_ratio") or 0)

    def ok(t: str) -> bool:
        n = len(t)
        if n < min_c or (max_c and n > max_c):
            return False
        w = len(t.split())
        if w < min_w or (max_w and w > max_w):
            return False
        if inc and not inc.search(t):
            return False
        if exc and exc.search(t):
            return False
        if max_non_alnum and n:
            sym = sum(1 for ch in t if not (ch.isalnum() or ch.isspace()))
            if sym / n > max_non_alnum:
                return False
        return True

    def fn(records):
        for r in records:
            if ok(record_text(r)):
                yield r
    return _rewrite(ds_id, fn, {"op": "filter", "options": opts})


def _norm_for_hash(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


_P = (1 << 31) - 1


def _minhash(t: str, a: np.ndarray, b: np.ndarray, k: int = 5) -> np.ndarray:
    words = re.findall(r"\w+", t.lower()) or [""]      # punctuation/whitespace-insensitive shingles
    if len(words) < k:
        shingles = {" ".join(words)}
    else:
        shingles = {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
    hs = np.fromiter((zlib.crc32(s.encode("utf-8")) % _P for s in shingles), dtype=np.int64, count=len(shingles))
    return ((np.outer(a, hs) + b[:, None]) % _P).min(axis=1)


def dedup(ds_id: str, opts: dict | None = None) -> dict:
    o = opts or {}
    near = bool(o.get("near", False))
    threshold = float(o.get("threshold", 0.85))
    perms, bands = 64, 16
    rows = perms // bands
    rng = np.random.default_rng(1234)
    a = rng.integers(1, _P, perms, dtype=np.int64)
    b = rng.integers(0, _P, perms, dtype=np.int64)
    seen: set[bytes] = set()
    buckets: dict[tuple, list[np.ndarray]] = {}

    def fn(records):
        for r in records:
            t = record_text(r)
            h = hashlib.blake2b(_norm_for_hash(t).encode("utf-8"), digest_size=16).digest()
            if h in seen:
                continue
            seen.add(h)
            if near:
                sig = _minhash(t, a, b)
                dup = False
                keys = [(i, sig[i * rows:(i + 1) * rows].tobytes()) for i in range(bands)]
                for key in keys:
                    for other in buckets.get(key, ()):
                        if float((other == sig).mean()) >= threshold:
                            dup = True
                            break
                    if dup:
                        break
                if dup:
                    continue
                for key in keys:
                    buckets.setdefault(key, []).append(sig)
            yield r
    return _rewrite(ds_id, fn, {"op": "dedup", "options": o})


def shuffle(ds_id: str, seed: int = 42) -> dict:
    def fn(records):
        items = list(records)
        random.Random(seed).shuffle(items)
        yield from items
    return _rewrite(ds_id, fn, {"op": "shuffle", "seed": seed})


# ================================================================ prepare
def _split_of(i: int, seed: int, val: float, test: float) -> str:
    x = (zlib.crc32(f"{seed}:{i}".encode()) & 0xFFFFFFFF) / 0xFFFFFFFF
    if x < test:
        return "test"
    if x < test + val:
        return "val"
    return "train"


def encode_record(r: dict, tok) -> tuple[list[int], list[int]]:
    """Token ids and loss mask (1 = learn this token) for one record."""
    eos = tok.eos_id
    bos = tok.bos_id
    if "text" in r:
        ids = ([bos] if bos is not None and tok.kind != "custom" else []) + tok.encode(r["text"])
        if eos is not None:
            ids.append(eos)
        return ids, [1] * len(ids)
    msgs = r["messages"]
    ids: list[int] = []
    mask: list[int] = []
    prev_text = ""
    for k in range(len(msgs)):
        m = msgs[k]
        if m["role"] == "assistant":
            prompt_text = tok.apply_chat_template(msgs[:k], add_generation_prompt=True)
            full_text = tok.apply_chat_template(msgs[:k + 1], add_generation_prompt=False)
            head = prompt_text[len(prev_text):] if prompt_text.startswith(prev_text) else prompt_text
            body = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            h = tok.encode(head)
            bd = tok.encode(body)
            ids += h + bd
            mask += [0] * len(h) + [1] * len(bd)
            prev_text = full_text
        else:
            full_text = tok.apply_chat_template(msgs[:k + 1], add_generation_prompt=False)
            seg = full_text[len(prev_text):] if full_text.startswith(prev_text) else full_text
            s = tok.encode(seg)
            ids += s
            mask += [0] * len(s)
            prev_text = full_text
    # Templates normally close assistant turns with their own end token (<|end|>, <|im_end|>...).
    # Only append EOS when the conversation does not already end on a stop token.
    if eos is not None and not any(t in ids[-3:] for t in tok.chat_stop_tokens()):
        ids.append(eos)
        mask.append(1)
    return ids, mask


def prepare(ds_id: str, tokenizer_dir: str, context_length: int, val_ratio: float = 0.02,
            test_ratio: float = 0.0, packing: bool = True, min_tokens: int = 1, max_tokens: int = 0,
            long_docs: str = "split", seed: int = 1337, progress: Callable[[int], None] | None = None) -> dict:
    """Tokenise a dataset into flat token + loss-mask streams for training."""
    from .tokenizer import load_tokenizer
    tok = load_tokenizer(tokenizer_dir)
    key = hashlib.sha1(f"{Path(tokenizer_dir).resolve()}|{context_length}|{val_ratio}|{test_ratio}|{packing}|"
                       f"{min_tokens}|{max_tokens}|{long_docs}|{seed}".encode()).hexdigest()[:12]
    out = ds_dir(ds_id) / "prepared" / key
    info_path = out / "prepared.json"
    meta = get_meta(ds_id)
    info = store.read_json(info_path)
    if info and info.get("dataset_updated") == meta.get("updated"):
        return info
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.uint32 if tok.vocab_size > 65535 else np.uint16
    pad = tok.pad_id if tok.pad_id is not None else (tok.eos_id or 0)
    row = context_length + 1
    files = {s: (open(out / f"{s}.tokens.bin", "wb"), open(out / f"{s}.mask.bin", "wb")) for s in ("train", "val", "test")}
    counts = {s: {"samples": 0, "tokens": 0, "trainable_tokens": 0, "max_len": 0, "dropped_short": 0,
                  "truncated": 0} for s in files}
    t0 = time.time()
    try:
        for i, r in enumerate(iter_records(ds_id)):
            split = _split_of(i, seed, val_ratio, test_ratio)
            ids, mask = encode_record(r, tok)
            c = counts[split]
            if len(ids) < max(1, min_tokens) or sum(mask) == 0:
                c["dropped_short"] += 1
                continue
            limit = max_tokens or 0
            if not packing:
                limit = min(limit, row) if limit else row
            pieces: list[tuple[list[int], list[int]]]
            if limit and len(ids) > limit:
                c["truncated"] += 1
                if long_docs == "split":
                    pieces = [(ids[j:j + limit], mask[j:j + limit]) for j in range(0, len(ids), limit)]
                else:
                    pieces = [(ids[:limit], mask[:limit])]
            else:
                pieces = [(ids, mask)]
            for pid, pm in pieces:
                if not packing and len(pid) < row:
                    pm = pm + [0] * (row - len(pid))
                    pid = pid + [pad] * (row - len(pid))
                ft, fm = files[split]
                ft.write(np.asarray(pid, dtype=dtype).tobytes())
                fm.write(np.asarray(pm, dtype=np.uint8).tobytes())
                c["samples"] += 1
                c["tokens"] += len(pid)
                c["trainable_tokens"] += int(sum(pm))
                c["max_len"] = max(c["max_len"], len(pid))
            if progress and i % 2000 == 0:
                progress(i)
    finally:
        for ft, fm in files.values():
            ft.close()
            fm.close()
    for s, c in counts.items():
        c["avg_len"] = round(c["tokens"] / c["samples"], 1) if c["samples"] else 0
        c["sequences"] = c["tokens"] // row
    info = {"key": key, "dataset": ds_id, "dataset_updated": meta.get("updated"), "tokenizer_dir": str(tokenizer_dir),
            "context_length": context_length, "packing": packing, "dtype": np.dtype(dtype).name,
            "vocab_size": tok.vocab_size, "splits": counts, "seconds": round(time.time() - t0, 2),
            "min_tokens": min_tokens, "max_tokens": max_tokens, "val_ratio": val_ratio, "test_ratio": test_ratio,
            "seed": seed, "path": str(out)}
    store.write_json(info_path, info)
    meta.setdefault("prepared", {})[key] = {"tokenizer_dir": str(tokenizer_dir), "context_length": context_length,
                                            "train_tokens": counts["train"]["tokens"],
                                            "val_tokens": counts["val"]["tokens"]}
    # Written directly (not via _save_meta) so 'updated' - the cache key - stays unchanged.
    store.write_json(store.datasets_dir() / ds_id / "meta.json", meta)
    return info


def iter_texts(ds_ids: list[str], limit_chars: int = 0) -> Iterator[str]:
    """Plain texts across datasets (chat records flattened) - used to train tokenizers."""
    total = 0
    for ds in ds_ids:
        for r in iter_records(ds):
            if "text" in r:
                t = r["text"]
            else:
                t = "\n".join(m["content"] for m in r["messages"])
            yield t
            total += len(t)
            if limit_chars and total >= limit_chars:
                return
