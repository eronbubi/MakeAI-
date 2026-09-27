"""API, sharing, projects, evaluation and the standalone (no Claude) guarantee."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT

H = {"X-MakeAI-Client": "1"}


@pytest.fixture(scope="module")
def client(trained):
    from makeai.server.app import create_app
    c = TestClient(create_app())
    c.patch("/api/settings", json={"profile": {"name": "Eron", "username": "eron"}}, headers=H)
    return c


def test_mutations_require_client_header(client):
    assert client.patch("/api/settings", json={}).status_code == 403
    assert client.post("/api/runs", json={}).status_code == 403


def test_playground_streams_locally(client, trained):
    chunks = []
    with client.stream("POST", "/api/playground/chat", headers=H, json={"uid": trained["uid"], "raw_prompt": "def ",
                                                                        "params": {"temperature": 0, "max_tokens": 12}}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                chunks.append(json.loads(line[6:]))
    assert any("delta" in c for c in chunks) and chunks[-1]["done"] and chunks[-1]["completion_tokens"] > 0


def test_sharing_visibility_and_public_page(client, trained):
    uid = trained["uid"]
    card = client.post(f"/api/models/{uid}/share", json={"visibility": "link"}, headers=H).json()
    page = client.get(card["page"].replace("http://testserver", ""))
    for s in ("EronAI", "Eron", "@eron", "A general-purpose AI", "Version", "Parameters", "Context", "coding",
              "Created", "Last update", "Required hardware", "Model format"):
        assert s in page.text, s
    assert not client.get("/api/discover").json()["items"]                 # link = not listed
    client.post(f"/api/models/{uid}/share", json={"visibility": "public"}, headers=H)
    items = client.get("/api/discover?category=coding").json()["items"]
    assert items and items[0]["creator"]["username"] == "eron"
    pkg = client.get(items[0]["download"].replace("http://testserver", ""))
    assert pkg.status_code == 200 and pkg.content[:2] == b"PK"
    assert client.get("/api/public/index").json()["items"][0]["downloads"] == 1
    client.post(f"/api/models/{uid}/share", json={"visibility": "private"}, headers=H)
    assert client.get(card["page"].replace("http://testserver", "")).status_code == 404


def test_evaluation_dataset_and_benchmark(client, trained):
    from makeai import evaluate
    r = evaluate.dataset_metrics(trained["uid"], trained["ds"], "val", 5)
    assert r["loss"] > 0 and r["perplexity"] > 1 and 0 <= r["token_accuracy"] <= 1
    from makeai.inference import InferenceManager
    b = evaluate.run_benchmark(trained["uid"], [{"prompt": "x", "expected": "zzzz_never", "match": "contains"},
                                                {"prompt": "y", "expected": "", "match": "contains"}], InferenceManager())
    assert b["n"] == 2 and b["accuracy"] == 0.5


def test_projects_contain_all_folders(client):
    p = client.post("/api/projects", json={"name": "Demo Lab"}, headers=H).json()
    tree = {t["path"] for t in client.get(f"/api/projects/{p['slug']}").json()["tree"]}
    for f in ("src", "models", "datasets", "tokenizers", "configs", "evaluation", "tests", "docs"):
        assert f in tree
    client.put(f"/api/projects/{p['slug']}/file", json={"path": "src/train.py", "content": "print(1)\n"}, headers=H)
    assert client.get(f"/api/projects/{p['slug']}/file?path=src/train.py").json()["content"] == "print(1)\n"
    assert client.put(f"/api/projects/{p['slug']}/file", json={"path": "../escape.txt", "content": "x"}, headers=H).status_code == 400


def test_runtime_never_imports_claude():
    """Static check: only makeai/devagent may reference Claude; the server imports it guardedly."""
    offenders = []
    for py in (ROOT / "makeai").rglob("*.py"):
        if "devagent" in py.parts or "claudemode" in py.parts:     # optional Claude integrations
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                if any(n and ("anthropic" in n or ("claude" in n.lower() and n not in ("claudemode",))) for n in names):
                    offenders.append(str(py))
    assert not offenders


def test_standalone_without_claude(tmp_path, trained):
    """Remove the development agent and the claude CLI; create, train, evaluate, run and share must still work."""
    script = f"""
import os, sys, shutil, json, time
sys.modules['makeai.devagent'] = None            # the agent package is gone
sys.modules['makeai.claudemode'] = None          # and so is Claude Mode
os.environ['PATH'] = ''                          # no claude CLI anywhere
os.environ['MAKEAI_HOME'] = {str(tmp_path)!r}
sys.path.insert(0, {str(ROOT)!r})
from fastapi.testclient import TestClient
from makeai.server.app import create_app
c = TestClient(create_app()); H = {{'X-MakeAI-Client': '1'}}
assert c.get('/api/dev/status').json()['available'] is False
c.patch('/api/settings', json={{'profile': {{'name': 'Ana', 'username': 'ana'}}}}, headers=H)
job = c.post('/api/import', json={{'path': {str(Path(os.environ['MAKEAI_HOME']) / 'exports' / 'eron--eronai--1.0' / 'eron--eronai--1.0.makeai')!r}}}, headers=H).json()
while c.get('/api/jobs/' + job['id']).json()['state'] == 'running': time.sleep(0.2)
assert c.get('/api/jobs/' + job['id']).json()['state'] == 'done', c.get('/api/jobs/' + job['id']).json()
ds = c.post('/api/datasets', json={{'name': 'd', 'paths': [{str(Path(os.__file__).parent / 'json')!r}]}}, headers=H).json()
while c.get('/api/jobs/' + ds['id']).json()['state'] == 'running': time.sleep(0.2)
ds_id = c.get('/api/jobs/' + ds['id']).json()['result']['id']
r = c.post('/api/runs', json={{'model_uid': 'eron--eronai--1.0', 'datasets': [{{'id': ds_id}}], 'training': {{'precision': 'fp32',
  'optimizer': 'adamw', 'scheduler': 'constant', 'learning_rate': 1e-4, 'micro_batch_size': 4, 'gradient_accumulation': 1,
  'context_length': 128, 'max_steps': 5, 'eval_every': 5, 'dataloader_workers': 1}}}}, headers=H).json()
while c.get('/api/runs/' + r['run_id']).json()['status'].get('state') not in ('completed', 'failed', 'crashed'): time.sleep(0.3)
assert c.get('/api/runs/' + r['run_id']).json()['status']['state'] == 'completed'
out = c.post('/api/playground/chat', json={{'uid': 'eron--eronai--1.0', 'raw_prompt': 'def ', 'params': {{'max_tokens': 5}}}}, headers=H).text
assert '"done": true' in out
assert c.post('/api/models/eron--eronai--1.0/share', json={{'visibility': 'public'}}, headers=H).status_code == 200
print('STANDALONE-OK')
"""
    from makeai.io import export as E
    E.export(trained["uid"], "package")
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert "STANDALONE-OK" in p.stdout, p.stdout[-2000:] + p.stderr[-3000:]
