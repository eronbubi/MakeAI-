"""LoRA / QLoRA / adapters / full fine-tuning on a model trained in this session."""
import glob
import json

import pytest
import torch

from conftest import QWEN, wait_state
from makeai import registry
from makeai.train.runs import RunManager, read_metrics

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _finetune(trained, method, **tr):
    import shutil
    base = trained["uid"]
    m = registry.create_model(name=f"Tuned-{method}-{tr.get('base_quant', 'bf16')}", creator_name="Max", creator_username="max", config=registry.get_config(base),
                              method=method, base_model=base)
    shutil.copytree(registry.tokenizer_dir(base), registry.tokenizer_dir(m["uid"]))
    training = {"precision": "bf16", "optimizer": "adamw", "scheduler": "cosine", "learning_rate": 1e-3 if method != "full" else 3e-4,
                "warmup_steps": 3, "micro_batch_size": 8, "gradient_accumulation": 1, "context_length": 128, "max_steps": 40,
                "eval_every": 20, "seed": 3, "base_quant": tr.pop("base_quant", "bf16"),
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "bias": "none", "double_quant": True,
                         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
                "adapter": {"bottleneck": 16}, **tr}
    rm = RunManager()
    rid = rm.create({"model_uid": m["uid"], "model_name": m["name"], "method": method, "base_model": base,
                     "tokenizer_dir": str(registry.tokenizer_dir(m["uid"])), "datasets": [{"id": trained["ds"], "weight": 1}],
                     "training": training, "checkpoint": {"save_every": 0}, "eval": {"max_batches": 5}})
    rm.start(rid)
    s = wait_state(rid, ("completed",), timeout=600)
    rm.shutdown()
    recs, _ = read_metrics(rid)
    return m["uid"], s, [r for r in recs if r["type"] == "step"]


@cuda
@pytest.mark.parametrize("method,quant", [("lora", "bf16"), ("qlora", "nf4"), ("qlora", "fp4"), ("adapter", "bf16"), ("full", "bf16")])
def test_parameter_efficient_methods_train_and_run(trained, method, quant):
    uid, status, steps = _finetune(trained, method, base_quant=quant)
    assert steps[-1]["loss"] == steps[-1]["loss"]          # finite
    if method != "full":
        assert status["trainable_params"] < status["params"] * 0.2
    log = (registry.store.runs_dir() / status_run(uid) / "log.txt").read_text(encoding="utf-8")
    if method == "qlora":
        assert f"to {quant.upper()}" in log
    m = registry.load_manifest(uid)
    assert m["runnable"] and m["original_creator"]["username"] == "max"
    from makeai.inference import InferenceManager
    im = InferenceManager()
    out = [c for c in im.stream(uid, [], {"temperature": 0, "max_tokens": 8}, raw_prompt="def ")][-1]
    assert out["completion_tokens"] > 0
    im.unload()


def status_run(uid):
    return registry.load_manifest(uid)["training_runs"][-1]


@cuda
def test_lora_export_is_peft_compatible(trained):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from makeai.io import export as E
    uid = registry.uid_for("max", "Tuned-lora-bf16", "1.0")
    if not (registry.store.models_dir() / uid).exists():
        uid, _, _ = _finetune(trained, "lora")
    hf = E.export(trained["uid"], "hf", {"dtype": "fp32"}).with_suffix("")
    ad = E.export(uid, "lora").with_suffix("")
    base = transformers.AutoModelForCausalLM.from_pretrained(str(hf), torch_dtype=torch.float32)
    pm = peft.PeftModel.from_pretrained(base, str(ad)).eval()
    ours, tok, _ = registry.load_model(uid, device="cpu", dtype=torch.float32)
    x = torch.tensor([tok.encode("import json\n")])
    with torch.no_grad():
        assert (ours(x)[0] - pm(x).logits).abs().max() < 1e-3


@cuda
def test_int8_base_on_aligned_model_and_clear_error_otherwise(trained):
    """LLM.int8 needs widths divisible by 32 (bitsandbytes). Misaligned models get an explanation, not a crash."""
    from conftest import tiny_config
    from makeai.model import MakeAIForCausalLM
    from makeai.model.peft import apply_lora, quantize_linears
    bad = MakeAIForCausalLM(tiny_config(512, n_kv_heads=2)).to(torch.bfloat16)      # k/v width 48
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_linears(bad, "int8", device="cuda")
    m = MakeAIForCausalLM(tiny_config(512, n_kv_heads=4)).to(torch.bfloat16)        # all widths multiples of 32
    quantize_linears(m, "int8", device="cuda")
    m.cuda()
    apply_lora(m, 8, 16, 0.0)
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-3)
    x = torch.randint(0, 512, (8, 64), device="cuda")
    losses = []
    for _ in range(25):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, labels=x)[1]
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
