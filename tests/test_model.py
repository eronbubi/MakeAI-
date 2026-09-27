import json

import pytest
import torch

from conftest import QWEN, tiny_config
from makeai.model import MakeAIForCausalLM
from makeai.model.generate import SamplingParams, generate_stream


@pytest.mark.parametrize("kw", [
    {}, {"activation": "gelu", "norm": "layernorm", "pos_encoding": "learned", "bias": True},
    {"pos_encoding": "alibi"}, {"attention_type": "sliding_window", "sliding_window": 16},
    {"embedding_size": 48, "tie_weights": False}, {"rope_scaling": "linear", "rope_scaling_factor": 2.0},
    {"activation": "geglu", "qkv_bias": True, "n_kv_heads": 1},
])
def test_architecture_variants_param_count_and_backward(kw):
    cfg = tiny_config(500, **kw)
    m = MakeAIForCausalLM(cfg)
    assert m.num_parameters() == cfg.param_count()
    x = torch.randint(0, 500, (2, 32))
    _, loss, _ = m(x, labels=x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@pytest.mark.parametrize("kw", [{}, {"pos_encoding": "alibi"}, {"attention_type": "sliding_window", "sliding_window": 8},
                                {"pos_encoding": "learned"}])
def test_kv_cache_matches_full_recompute(kw):
    torch.manual_seed(0)
    cfg = tiny_config(300, **kw)
    m = MakeAIForCausalLM(cfg).eval()
    x = torch.randint(0, 300, (1, 20))
    with torch.no_grad():
        full = m(x)[0][0, -1]
        _, _, cache = m(x[:, :-1], use_cache=True)
        inc = m(x[:, -1:], past=cache, use_cache=True)[0][0, -1]
    assert torch.allclose(full, inc, atol=1e-4)


class _Tok:
    def decode(self, ids, skip_special=True):
        return "".join(chr(97 + i % 26) for i in ids)


def test_sampling_controls():
    torch.manual_seed(0)
    cfg = tiny_config(200)
    m = MakeAIForCausalLM(cfg).eval()
    run = lambda **kw: [c for c in generate_stream(m, _Tok(), [1, 2, 3], SamplingParams(**kw))][-1]
    g1, g2 = run(temperature=0, max_tokens=10), run(temperature=0, max_tokens=10)
    assert g1["text"] == g2["text"] and g1["completion_tokens"] == 10          # greedy is deterministic
    s1, s2 = run(temperature=1.0, seed=5, max_tokens=12), run(temperature=1.0, seed=5, max_tokens=12)
    assert s1["text"] == s2["text"]                                            # seed reproduces sampling
    assert run(temperature=0, max_tokens=4)["completion_tokens"] == 4          # max tokens honoured
    stop = g1["text"][3:5]
    r = run(temperature=0, max_tokens=10, stop=[stop])
    assert r["finish_reason"] == "stop_sequence" and stop not in r["text"]


@pytest.mark.slow
@pytest.mark.skipif(not QWEN, reason="Qwen2.5-0.5B not in the Hugging Face cache")
def test_hf_qwen2_weights_run_natively_with_identical_logits():
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from makeai.model.hf_compat import from_hf_config
    snap = QWEN[0]
    cfg, _ = from_hf_config(json.load(open(snap + "/config.json")))
    m = MakeAIForCausalLM(cfg)
    m.load_state_dict(load_file(snap + "/model.safetensors"), strict=False)
    m = m.cuda().eval()
    ids = torch.randint(0, 150000, (1, 40)).cuda()
    with torch.no_grad():
        ours = m(ids)[0].float().cpu()
    del m
    ref = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        theirs = ref(ids).logits.float().cpu()
    assert (ours - theirs).abs().max().item() < 1e-3
