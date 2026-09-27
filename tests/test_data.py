import json

import numpy as np
import pytest

from makeai.data import datasets as D
from makeai.data.tokenizer import load_tokenizer, train_tokenizer

SAMPLE = [
    {"text": "def add(a, b):\n    return a + b\n"},
    {"messages": [{"role": "user", "content": "Say hi"}, {"role": "assistant", "content": "Hi!"}]},
    {"instruction": "Reverse 'abc'", "output": "cba"},
    {"prompt": "2+2?", "response": "4"},
]


def test_import_all_formats(home, tmp_path):
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in SAMPLE), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(SAMPLE), encoding="utf-8")
    (tmp_path / "c.txt").write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    (tmp_path / "d.csv").write_text("text\nhello world\nsecond row\n", encoding="utf-8")
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"content": ["parquet one", "parquet two"]}), tmp_path / "e.parquet")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "x.py").write_text("print('x')\n")
    (folder / "y.md").write_text("# y\n")
    m = D.create_dataset("all", [str(tmp_path / f) for f in ("a.jsonl", "b.json", "c.txt", "d.csv", "e.parquet")] + [str(folder)],
                         {"txt_mode": "paragraph"})
    s = m["stats"]
    assert s["samples"] == 4 + 4 + 2 + 2 + 2 + 2
    assert s["kinds"]["chat"] == 6       # messages + instruction/output + prompt/response, twice
    rows = D.preview(m["id"], 0, 100)
    assert any("messages" in r for r in rows) and any(r.get("text") == "parquet two" for r in rows)


def test_cleaning_filtering_dedup_shuffle(home):
    recs = [{"text": "Hello   world\x00  \r\n\r\n\r\n\r\n\r\nend  "}, {"text": "Hello   world\x00  \r\n\r\n\r\n\r\n\r\nend  "},
            {"text": "short"}, {"text": "The quick brown fox jumps over the lazy dog near the river bank today"},
            {"text": "The quick brown fox jumps over the lazy dog near the river bank today!"},
            {"text": " ".join(f"word{i}" for i in range(200))},
            {"text": " ".join(f"word{i}" if i != 100 else "changed" for i in range(200))}]
    m = D.create_from_records("tools", recs)
    D.clean(m["id"])
    r = D.preview(m["id"])[0]["text"]
    assert "\x00" not in r and "\r" not in r and "\n\n\n\n" not in r and not r.endswith(" ")
    assert D.dedup(m["id"])["removed"] == 1                   # exact duplicate
    assert D.dedup(m["id"], {"near": True})["removed"] == 2   # near duplicates: '!' apart, 1 of 200 words changed
    assert D.filter_records(m["id"], {"min_chars": 10})["removed"] == 1
    before = [x["text"] for x in D.preview(m["id"])]
    D.shuffle(m["id"], seed=1)
    after = [x["text"] for x in D.preview(m["id"])]
    assert sorted(before) == sorted(after)


@pytest.mark.parametrize("kind", ["bpe", "wordpiece", "unigram", "sentencepiece"])
def test_tokenizer_types(home, corpus, kind):
    tok = train_tokenizer(kind, D.iter_texts([corpus["ds"]], 400_000), 1024, home / "tokenizers" / f"t-{kind}")
    text = "def main(argv):\n    return json.dumps({'a': 1})\n"
    ids = tok.encode(text)
    assert 0 < len(ids) < len(text)
    if kind != "wordpiece":                     # WordPiece does not preserve whitespace by design
        assert tok.decode(ids) == text
    for role in ("bos", "eos", "pad", "unk"):
        assert tok.special_id(role) is not None
    again = load_tokenizer(home / "tokenizers" / f"t-{kind}")
    assert again.encode(text) == ids


def test_prepare_split_packing_and_chat_mask(home, corpus):
    tok = load_tokenizer(corpus["tok_dir"])
    ids, mask = D.encode_record({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}, tok)
    assert mask[0] == 0 and mask[-1] == 1 and 0 < sum(mask) < len(mask)   # only the reply is learned
    info = D.prepare(corpus["ds"], str(corpus["tok_dir"]), 64, val_ratio=0.1, packing=True)
    tr = np.fromfile(f"{info['path']}/train.tokens.bin", dtype=info["dtype"])
    assert len(tr) == info["splits"]["train"]["tokens"] and info["splits"]["val"]["samples"] > 0
    stats = D.compute_stats(corpus["ds"], tok)
    assert stats["tokens"] > 0 and stats["max_tokens"] >= stats["avg_tokens"]
    unpacked = D.prepare(corpus["ds"], str(corpus["tok_dir"]), 64, val_ratio=0.1, packing=False, min_tokens=10)
    assert unpacked["splits"]["train"]["max_len"] == 65                    # one padded row per sample
