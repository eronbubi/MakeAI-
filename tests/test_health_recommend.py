from makeai.hardware.recommend import COMPLEXITY, recommend
from makeai.model.config import ModelConfig
from makeai.train import health

HW = {"best_device": {"type": "cuda", "index": 0, "name": "GPU", "vram_mb": 8192, "bf16": True},
      "gpus": [{"index": 0, "vram_used_mb": 500}], "ram": {"total_mb": 16000}}


def _tel(util, temp=60, vram=6000, throttle=(), ram=60):
    return [{"t": i, "gpus": [{"util_pct": util, "temp_c": temp, "vram_used_mb": vram, "vram_total_mb": 8192,
                               "throttle": list(throttle)}], "ram_pct": ram} for i in range(10)]


def _steps(tps=50_000, wait=0.01, losses=None):
    losses = losses or [3.0 - 0.01 * i for i in range(20)]
    return [{"loss": l, "tokens_per_s": tps, "data_wait_frac": wait, "mem_reserved_mb": 5500, "grad_norm": 0.8,
             "errors": 0} for l in losses]


def score(**kw):
    tel = _tel(kw.get("util", 95), kw.get("temp", 60), throttle=kw.get("throttle", ()), ram=kw.get("ram", 60))
    st = _steps(kw.get("tps", 50_000), kw.get("wait", 0.01), kw.get("losses"))
    return health.compute(st, tel, gpu_static={"vram_total_mb": 8192, "slowdown_temp_c": 93},
                          run_cfg={"training": {"dataloader_workers": 4, "micro_batch_size": 8, "learning_rate": 1e-3, "grad_clip": 1.0}},
                          flops_per_token=6 * 50e6, peak_flops=30e12)


def test_performance_score_reacts_to_measured_conditions():
    good = score()
    assert good["available"] and 1 <= good["score"] <= 10 and good["status"] in ("excellent", "good")
    assert score(util=40)["score"] < good["score"]                      # utilisation
    assert score(tps=10_000)["score"] < good["score"]                   # throughput
    hot = score(temp=91, throttle=("hw_thermal",))
    assert hot["score"] < good["score"] and hot["status"] == "critical"  # thermals
    data = score(wait=0.4, util=55)
    assert data["score"] < good["score"] and "DataLoader workers from 4 → 8" in data["recommendation"]
    spike = score(losses=[2.0] * 17 + [2.0, 2.1, 6.0])
    assert spike["components"]["stability"]["score"] < good["components"]["stability"]["score"]
    assert "not a measure of model quality" in good["note"]


def test_score_skips_components_without_measurements():
    h = health.compute(_steps(), [], gpu_static={}, run_cfg={})
    assert h["available"] and "gpu_utilization" not in h["components"] and "thermals" not in h["components"]


def test_recommendations_scale_with_complexity_and_fit():
    prev = 0
    for c in COMPLEXITY:
        r = recommend(HW, c, "from_scratch")
        assert r["estimates"]["fits"], c
        assert r["estimates"]["params"] >= prev * 0.9
        prev = r["estimates"]["params"]
        t = r["training"]
        for k in ("precision", "optimizer", "learning_rate", "micro_batch_size", "gradient_accumulation", "max_steps",
                  "context_length", "warmup_steps"):
            assert t[k] is not None
        assert r["model"]["n_heads"] % r["model"]["n_kv_heads"] == 0
        assert r["estimates"]["duration_s"] > 0 and r["dataset"]["recommended_tokens"] > 0
    assert recommend(HW, 1, "from_scratch")["training"]["precision"] == "bf16"


def test_qlora_recommendation_for_base_model():
    base = ModelConfig(vocab_size=151936, n_layers=24, hidden_size=896, intermediate_size=4864, n_heads=14, n_kv_heads=2,
                       head_dim=64, context_length=32768, qkv_bias=True)
    r = recommend(HW, 16, "qlora", base_config=base)
    assert r["training"]["base_quant"] == "nf4" and r["training"]["lora"]["rank"] == 32
    assert r["estimates"]["trainable_params"] < base.param_count() * 0.05
