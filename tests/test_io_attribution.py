"""Export/import in every format; the original creator must survive every round trip."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from conftest import switch_home
from makeai import registry
from makeai.inference import InferenceManager
from makeai.io import export as E
from makeai.io.importer import import_model, inspect


def greedy(uid, n=16):
    im = InferenceManager()
    out = "".join(c.get("delta", "") for c in im.stream(uid, [], {"temperature": 0, "max_tokens": n, "repetition_penalty": 1.0},
                                                         raw_prompt="def "))
    im.unload()
    return out


def test_original_creator_survives_export_import_chain(trained, home):
    uid = trained["uid"]
    ref = greedy(uid)
    paths = {f: E.export(uid, f) for f in ("package", "safetensors", "pytorch", "onnx", "gguf")}
    homes = [Path(tempfile.mkdtemp(prefix="makeai-max-")), Path(tempfile.mkdtemp(prefix="makeai-lena-"))]
    try:
        switch_home(homes[0])
        for fmt, p in paths.items():
            m = import_model(str(p), importer={"name": "Max", "username": "max"})
            assert m["creator"] == {"name": "Eron", "username": "eron"}, fmt
            assert m["original_creator"]["username"] == "eron", fmt
            assert [i["username"] for i in m["imported_by"]] == ["max"], fmt
            assert m["id"] == "makeai://eron/eronai/1.0"
            if fmt in ("package", "safetensors", "pytorch", "gguf", "onnx"):
                assert greedy(m["uid"]) == ref, fmt                     # same model, same output
            registry.delete_model(m["uid"])
        import_model(str(paths["package"]), importer={"name": "Max", "username": "max"})
        registry.update_model("eron--eronai--1.0", {"creator": {"name": "Max", "username": "max"}})
        assert registry.load_manifest("eron--eronai--1.0")["creator"]["username"] == "eron"   # cannot be replaced
        again = E.export("eron--eronai--1.0", "package")
        switch_home(homes[1])
        m = import_model(str(again), importer={"name": "Lena", "username": "lena"})
        assert m["original_creator"]["username"] == "eron"
        assert [i["username"] for i in m["imported_by"]] == ["max", "lena"]
    finally:
        switch_home(home)
        for h in homes:
            shutil.rmtree(h, ignore_errors=True)


def test_onnx_export_matches_pytorch(trained):
    p = E.export(trained["uid"], "onnx")      # export_onnx verifies |logit diff| < 1e-2 against PyTorch itself
    assert p.exists() and inspect(str(p))["runnable"]


def test_gguf_runs_in_llama_cpp_identically(trained):
    llama_cpp = pytest.importorskip("llama_cpp")
    g = E.export(trained["uid"], "gguf")
    llm = llama_cpp.Llama(model_path=str(g), n_ctx=128, verbose=False)
    out = llm.create_completion("def ", temperature=0, max_tokens=16, repeat_penalty=1.0)["choices"][0]["text"]
    assert out == greedy(trained["uid"])
    info = inspect(str(g))
    assert info["attribution"]["original_creator"]["username"] == "eron" and info["author"] == "Eron (@eron)"


def test_hf_folder_loads_in_transformers(trained):
    transformers = pytest.importorskip("transformers")
    import torch
    z = E.export(trained["uid"], "hf", {"dtype": "fp32"})
    m = transformers.AutoModelForCausalLM.from_pretrained(str(z.with_suffix("")), torch_dtype=torch.float32).eval()
    ours, tok, _ = registry.load_model(trained["uid"], device="cpu", dtype=torch.float32)
    x = torch.tensor([tok.encode("import os\n")])
    with torch.no_grad():
        assert (ours(x)[0] - m(x).logits).abs().max() < 1e-3


def test_tensorrt_reported_honestly(trained):
    av = E.availability(trained["uid"])["tensorrt"]
    try:
        import tensorrt  # noqa: F401
    except Exception:
        assert not av["available"] and "TensorRT" in av["reason"]


def test_ai_id_format(trained):
    m = registry.load_manifest(trained["uid"])
    assert m["id"] == "makeai://eron/eronai/1.0"
    assert registry.parse_ai_id(m["id"]) == {"username": "eron", "slug": "eronai", "version": "1.0"}
