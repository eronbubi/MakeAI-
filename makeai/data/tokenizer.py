"""Tokenizers: training (BPE, WordPiece, Unigram, SentencePiece), loading, chat templates.

A MakeAI tokenizer directory contains ``makeai_tokenizer.json`` (kind, special
tokens, chat template) plus either ``tokenizer.json`` (Hugging Face
``tokenizers`` format) or ``spm.model`` (SentencePiece).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator

KINDS = ("bpe", "wordpiece", "unigram", "sentencepiece", "custom")

DEFAULT_SPECIALS = {"pad": "<|pad|>", "bos": "<|bos|>", "eos": "<|eos|>", "unk": "<|unk|>"}
CHAT_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

# Chat format used by models created in MakeAI. Rendered with Jinja2 exactly like
# Hugging Face chat templates, so imported models keep their own template.
MAKEAI_CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


class MakeAITokenizer:
    def __init__(self, kind: str, backend: Any, specials: dict[str, str], extra_specials: list[str],
                 chat_template: str | None, meta: dict[str, Any] | None = None):
        self.kind = kind
        self.backend = backend            # tokenizers.Tokenizer or sentencepiece.SentencePieceProcessor
        self.specials = specials          # role -> token string (pad/bos/eos/unk)
        self.extra_specials = extra_specials
        self.chat_template = chat_template
        self.meta = meta or {}
        self._is_spm = kind == "sentencepiece"

    # ----------------------------------------------------------------- basics
    @property
    def vocab_size(self) -> int:
        if self._is_spm:
            return self.backend.get_piece_size()
        return self.backend.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, tok: str | None) -> int | None:
        if tok is None:
            return None
        if self._is_spm:
            i = self.backend.piece_to_id(tok)
            return None if i == self.backend.unk_id() and tok != self.backend.id_to_piece(i) else i
        return self.backend.token_to_id(tok)

    def id_to_token(self, i: int) -> str:
        return self.backend.id_to_piece(i) if self._is_spm else (self.backend.id_to_token(i) or "")

    def special_id(self, role: str) -> int | None:
        return self.token_to_id(self.specials.get(role))

    @property
    def eos_id(self) -> int | None: return self.special_id("eos")
    @property
    def bos_id(self) -> int | None: return self.special_id("bos")
    @property
    def pad_id(self) -> int | None: return self.special_id("pad")

    def all_special_ids(self) -> set[int]:
        out = set()
        for t in list(self.specials.values()) + list(self.extra_specials):
            i = self.token_to_id(t)
            if i is not None:
                out.add(i)
        return out

    def encode(self, text: str) -> list[int]:
        if self._is_spm:
            return self._spm_encode(text)
        return self.backend.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self._is_spm:
            return [self._spm_encode(t) for t in texts]
        return [e.ids for e in self.backend.encode_batch(texts, add_special_tokens=False)]

    def _spm_encode(self, text: str) -> list[int]:
        # SentencePiece does not know our special strings; split on them first.
        specials = [t for t in list(self.specials.values()) + self.extra_specials if t and t in text]
        if not specials:
            return self.backend.encode(text)
        out: list[int] = []
        import re
        pattern = "(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")"
        for part in re.split(pattern, text):
            if not part:
                continue
            if part in specials:
                out.append(self.backend.piece_to_id(part))
            else:
                out.extend(self.backend.encode(part))
        return out

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if self._is_spm:
            if skip_special:
                sp = self.all_special_ids()
                ids = [i for i in ids if i not in sp]
            return self.backend.decode(ids)
        return self.backend.decode(ids, skip_special_tokens=skip_special)

    # ------------------------------------------------------------------- chat
    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        template = self.chat_template or MAKEAI_CHAT_TEMPLATE
        import jinja2
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
        env.globals["raise_exception"] = _raise
        tpl = env.from_string(template)
        return tpl.render(messages=messages, add_generation_prompt=add_generation_prompt,
                          bos_token=self.specials.get("bos") or "", eos_token=self.specials.get("eos") or "")

    def chat_stop_tokens(self) -> list[int]:
        cands = [self.specials.get("eos"), "<|end|>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>", "</s>"]
        out = []
        for c in cands:
            i = self.token_to_id(c) if c else None
            if i is not None and i not in out:
                out.append(i)
        return out

    # ------------------------------------------------------------------- save
    def save(self, directory: Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self._is_spm:
            src = self.meta.get("spm_path")
            if src and Path(src).resolve() != (d / "spm.model").resolve():
                shutil.copyfile(src, d / "spm.model")
            self.meta["spm_path"] = str(d / "spm.model")
        else:
            self.backend.save(str(d / "tokenizer.json"))
        meta = {k: v for k, v in self.meta.items() if k != "spm_path"}
        info = {"kind": self.kind, "specials": self.specials, "extra_specials": self.extra_specials,
                "chat_template": self.chat_template, "vocab_size": self.vocab_size, "meta": meta}
        (d / "makeai_tokenizer.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "vocab_size": self.vocab_size, "specials": self.specials,
                "special_ids": {k: self.special_id(k) for k in self.specials},
                "extra_specials": self.extra_specials, "has_chat_template": bool(self.chat_template),
                "meta": {k: v for k, v in self.meta.items() if k != "spm_path"}}


def _raise(msg):
    raise ValueError(msg)


# ======================================================================= load
def load_tokenizer(directory: str | os.PathLike) -> MakeAITokenizer:
    d = Path(directory)
    info_path = d / "makeai_tokenizer.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        kind = info["kind"]
        if kind == "sentencepiece":
            import sentencepiece as spm
            backend = spm.SentencePieceProcessor(model_file=str(d / "spm.model"))
            meta = dict(info.get("meta") or {})
            meta["spm_path"] = str(d / "spm.model")
        else:
            from tokenizers import Tokenizer
            backend = Tokenizer.from_file(str(d / "tokenizer.json"))
            meta = info.get("meta") or {}
        return MakeAITokenizer(kind, backend, info["specials"], info.get("extra_specials", []),
                               info.get("chat_template"), meta)
    return load_hf_tokenizer(d)


def load_hf_tokenizer(d: Path) -> MakeAITokenizer:
    """Load a Hugging Face tokenizer directory (tokenizer.json + tokenizer_config.json)."""
    from tokenizers import Tokenizer
    if (d / "tokenizer.json").exists():
        backend = Tokenizer.from_file(str(d / "tokenizer.json"))
        kind = "custom"
    elif (d / "tokenizer.model").exists():
        import sentencepiece as spm
        backend = spm.SentencePieceProcessor(model_file=str(d / "tokenizer.model"))
        kind = "sentencepiece"
    else:
        raise FileNotFoundError(f"No tokenizer.json or tokenizer.model in {d}")
    cfg = {}
    if (d / "tokenizer_config.json").exists():
        cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))

    def tok_str(v):
        if isinstance(v, dict):
            return v.get("content")
        return v

    specials = {}
    for role in ("bos", "eos", "pad", "unk"):
        s = tok_str(cfg.get(f"{role}_token"))
        if s:
            specials[role] = s
    extra = []
    for v in (cfg.get("added_tokens_decoder") or {}).values():
        if v.get("special") and v.get("content") not in specials.values():
            extra.append(v["content"])
    template = cfg.get("chat_template")
    if isinstance(template, list):  # multiple named templates
        template = next((t["template"] for t in template if t.get("name") == "default"), template[0]["template"])
    meta = {"source": "huggingface"}
    if kind == "sentencepiece":
        meta["spm_path"] = str(d / "tokenizer.model")
    if kind == "custom" and not specials.get("eos"):
        for cand in ("<|endoftext|>", "</s>", "<|end_of_text|>"):
            if backend.token_to_id(cand) is not None:
                specials["eos"] = cand
                break
    return MakeAITokenizer(kind, backend, specials, extra, template, meta)


# ====================================================================== train
def train_tokenizer(kind: str, texts: Iterable[str], vocab_size: int, out_dir: Path,
                    specials: dict[str, str] | None = None, extra_specials: list[str] | None = None,
                    min_frequency: int = 2, chat_template: str | None = MAKEAI_CHAT_TEMPLATE,
                    spm_model_type: str = "unigram", character_coverage: float = 0.9995) -> MakeAITokenizer:
    """Train a new tokenizer on an iterator of texts and save it to ``out_dir``."""
    if kind not in KINDS or kind == "custom":
        raise ValueError(f"trainable kinds: bpe, wordpiece, unigram, sentencepiece")
    specials = {**DEFAULT_SPECIALS, **(specials or {})}
    extra = list(dict.fromkeys((extra_specials if extra_specials is not None else CHAT_TOKENS)))
    ordered = [specials["pad"], specials["bos"], specials["eos"], specials["unk"]] + [e for e in extra if e not in specials.values()]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "sentencepiece":
        import sentencepiece as spm
        corpus = out_dir / "_corpus.txt"
        with open(corpus, "w", encoding="utf-8") as f:
            for t in texts:
                t = t.replace("\r\n", "\n")
                for line in t.split("\n"):
                    if line.strip():
                        f.write(line + "\n")
        prefix = str(out_dir / "spm")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=prefix, vocab_size=vocab_size, model_type=spm_model_type,
            character_coverage=character_coverage, byte_fallback=True,
            pad_id=0, bos_id=1, eos_id=2, unk_id=3,
            pad_piece=specials["pad"], bos_piece=specials["bos"], eos_piece=specials["eos"], unk_piece=specials["unk"],
            user_defined_symbols=[e for e in extra if e not in specials.values()],
            input_sentence_size=2_000_000, shuffle_input_sentence=True, train_extremely_large_corpus=False,
            num_threads=os.cpu_count() or 4, minloglevel=2, remove_extra_whitespaces=False,
            normalization_rule_name="identity", split_digits=True, allow_whitespace_only_pieces=True,
        )
        corpus.unlink(missing_ok=True)
        (out_dir / "spm.vocab").unlink(missing_ok=True)
        import sentencepiece as spm2
        backend = spm2.SentencePieceProcessor(model_file=prefix + ".model")
        tok = MakeAITokenizer("sentencepiece", backend, specials, extra, chat_template,
                              {"spm_path": prefix + ".model", "model_type": spm_model_type})
        tok.save(out_dir)
        return tok

    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    if kind == "bpe":
        tk = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tk.decoder = decoders.ByteLevel()
        tk.post_processor = processors.ByteLevel(trim_offsets=False)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=ordered,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    elif kind == "wordpiece":
        tk = Tokenizer(models.WordPiece(unk_token=specials["unk"]))
        tk.normalizer = normalizers.NFC()
        tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tk.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                            special_tokens=ordered, show_progress=False)
    else:  # unigram
        tk = Tokenizer(models.Unigram())
        tk.normalizer = normalizers.NFC()
        tk.pre_tokenizer = pre_tokenizers.Metaspace()
        tk.decoder = decoders.Metaspace()
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=ordered, unk_token=specials["unk"],
                                          show_progress=False)
    tk.train_from_iterator(_batched(texts), trainer=trainer)
    tok = MakeAITokenizer(kind, tk, specials, extra, chat_template, {"min_frequency": min_frequency})
    tok.save(out_dir)
    return tok


def _batched(texts: Iterable[str], n: int = 256) -> Iterator[list[str]]:
    batch: list[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
