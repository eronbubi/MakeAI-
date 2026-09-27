"""Local text generation: KV cache, sampling controls, stop sequences, streaming."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    repetition_window: int = 256
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 256
    seed: int | None = None
    stop: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SamplingParams":
        names = cls.__dataclass_fields__
        p = cls(**{k: v for k, v in d.items() if k in names and v is not None})
        if isinstance(p.stop, str):
            p.stop = [p.stop] if p.stop else []
        p.stop = [s for s in p.stop if s]
        return p


def process_logits(logits: torch.Tensor, ctx_ids: list[int], gen_counts: dict[int, int], p: SamplingParams) -> torch.Tensor:
    """Apply penalties and filters to a 1-D float32 logits vector. Returns filtered logits (-inf = removed)."""
    logits = logits.float().clone()
    if p.repetition_penalty and p.repetition_penalty != 1.0 and ctx_ids:
        window = ctx_ids[-p.repetition_window:] if p.repetition_window > 0 else ctx_ids
        idx = torch.tensor(sorted(set(window)), device=logits.device, dtype=torch.long)
        vals = logits[idx]
        logits[idx] = torch.where(vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty)
    if (p.frequency_penalty or p.presence_penalty) and gen_counts:
        idx = torch.tensor(list(gen_counts.keys()), device=logits.device, dtype=torch.long)
        cnt = torch.tensor(list(gen_counts.values()), device=logits.device, dtype=torch.float32)
        logits[idx] -= p.frequency_penalty * cnt + p.presence_penalty
    if p.temperature <= 0:
        return logits
    logits = logits / p.temperature
    if p.top_k and p.top_k > 0 and p.top_k < logits.numel():
        kth = torch.topk(logits, p.top_k).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if p.min_p and p.min_p > 0:
        logits[probs < p.min_p * probs.max()] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
    if p.top_p and 0 < p.top_p < 1:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        remove = cum - sp > p.top_p          # keep the smallest set with mass >= top_p
        logits[si[remove]] = float("-inf")
    return logits


def sample(logits: torch.Tensor, p: SamplingParams, gen: torch.Generator | None) -> int:
    if p.temperature <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


class StreamDecoder:
    """Incremental detokenisation that never emits half of a multi-byte character."""

    def __init__(self, tok):
        self.tok = tok
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, i: int) -> str:
        self.ids.append(i)
        text = self.tok.decode(self.ids, skip_special=True)
        if text.endswith("�"):
            return ""
        delta = text[len(self.emitted):] if text.startswith(self.emitted) else text[len(self.emitted):]
        self.emitted = text
        return delta


@torch.inference_mode()
def generate_stream(model, tok, prompt_ids: list[int], params: SamplingParams, *,
                    stop_token_ids: list[int] | None = None,
                    should_stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """Yield {'delta': str} chunks, then a final {'done': True, stats...} record."""
    device = next(model.parameters()).device
    ctx_len = model.cfg.context_length
    max_new = max(1, int(params.max_tokens))
    if len(prompt_ids) + 1 > ctx_len:
        prompt_ids = prompt_ids[-(ctx_len - 1):]
    gen = None
    if params.seed is not None and params.temperature > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(params.seed))
    stop_ids = set(stop_token_ids or [])
    dec = StreamDecoder(tok)
    ids = list(prompt_ids)
    counts: dict[int, int] = {}
    t0 = time.perf_counter()
    x = torch.tensor([prompt_ids], device=device)
    logits, _, cache = model(x, use_cache=True, last_only=True)
    t_first = None
    finish = "length"
    text = ""
    n_gen = 0
    for _ in range(max_new):
        lg = process_logits(logits[0, -1], ids, counts, params)
        nxt = sample(lg, params, gen)
        if t_first is None:
            t_first = time.perf_counter() - t0
        if nxt in stop_ids:
            finish = "stop_token"
            break
        ids.append(nxt)
        counts[nxt] = counts.get(nxt, 0) + 1
        n_gen += 1
        delta = dec.push(nxt)
        if delta:
            text += delta
            hit = next((s for s in params.stop if s in text[-(len(delta) + max(len(s) for s in params.stop)):]), None) \
                if params.stop else None
            if hit:
                cut = text.find(hit)
                keep = delta[: max(0, len(delta) - (len(text) - cut))]
                if keep:
                    yield {"delta": keep}
                text = text[:cut]
                finish = "stop_sequence"
                break
            yield {"delta": delta}
        if should_stop and should_stop():
            finish = "cancelled"
            break
        if len(ids) >= ctx_len:
            finish = "context_full"
            break
        logits, _, cache = model(torch.tensor([[nxt]], device=device), past=cache, use_cache=True)
    total = time.perf_counter() - t0
    decode_time = total - (t_first or 0)
    yield {"done": True, "finish_reason": finish, "prompt_tokens": len(prompt_ids), "completion_tokens": n_gen,
           "time_to_first_token_s": round(t_first or 0, 4), "total_s": round(total, 3),
           "tokens_per_s": round((n_gen - 1) / decode_time, 1) if n_gen > 1 and decode_time > 0 else None,
           "text": text}


def build_chat_prompt(tok, messages: list[dict[str, str]], system_prompt: str | None, ctx_len: int,
                      max_new: int) -> tuple[list[int], int]:
    """Render with the model's chat template; drop the oldest turns until prompt + reply fit the context."""
    msgs = [m for m in messages if m.get("role") in ("user", "assistant", "system")]
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + [m for m in msgs if m["role"] != "system"]
    dropped = 0
    while True:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True)
        ids = tok.encode(text)
        bos = tok.bos_id
        if bos is not None and tok.kind != "custom" and (not ids or ids[0] != bos) and not text.startswith(tok.specials.get("bos") or "\0"):
            ids = [bos] + ids
        if len(ids) + max_new <= ctx_len:
            return ids, dropped
        body = [i for i, m in enumerate(msgs) if m["role"] != "system"]
        if len(body) <= 1:
            return ids[-(ctx_len - max_new):] if ctx_len > max_new else ids[-(ctx_len - 1):], dropped
        del msgs[body[0]]
        dropped += 1
