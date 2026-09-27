# MakeAI — by Convergent

MakeAI is a local AI laboratory: create language models from scratch or fine-tune existing ones, train them on your own GPU with a live hardware dashboard, evaluate them, chat with them, and export or share them — all on your computer. The runtime needs no cloud service and no Claude. The optional development agent (`makeai/devagent`) only helps develop MakeAI itself.

```
OPEN MAKEAI → HARDWARE SCAN → DASHBOARD → CREATE AI → NAME + CREATOR → ARCHITECTURE → COMPLEXITY
→ DATASET → AUTOMATIC HARDWARE OPTIMIZATION → REVIEW → START TRAINING → LIVE MONITORING
→ PAUSE / RESUME / KILL → EVALUATION → MY AIs → RUN → CHAT → SHARE / EXPORT
```

## Install (Windows)

Download **MakeAI-Setup.exe** from the [latest release](https://github.com/eronbubi/MakeAI-/releases/latest) and run it. Setup installs Python 3.10 and PyTorch (CUDA build when an NVIDIA GPU is present) into its own folder, adds MakeAI to the desktop and Start menu, and starts it. Build the installer yourself with `python installer/build.py`.

## Start (from source)

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run.py
```

**Desktop app (Windows):** `.venv\Scripts\python -m makeai.desktop` puts a MakeAI icon on the desktop. Double-click it and MakeAI starts without a console in its own app window (Chrome/Edge `--app`); closing the window stops MakeAI unless a training run is still going. A second double-click only opens another window. Output goes to `~/MakeAI/makeai.log`.

MakeAI opens at `http://127.0.0.1:7860`. Data lives in `~/MakeAI` (change with `--home DIR` or `MAKEAI_HOME`). `--host 0.0.0.0` lets other computers open shared AI pages; everything except the public share pages stays local-only. Optional: `pip install llama-cpp-python` to run imported GGUF files.

The UI is prebuilt into `makeai/web`. To change it: `cd ui && npm install && npm run build` (or `npm run dev` for hot reload against a running server).

## What is inside

| Part | Where | What it does |
|---|---|---|
| Hardware scanner | `makeai/hardware/scanner.py` | GPU (NVML), VRAM, temperature, power, clocks, throttle reasons, compute capability, FP16/BF16/TF32, Tensor Cores, CUDA/ROCm, multi-GPU, CPU, RAM, storage, OS. Live telemetry every second. Values the OS does not expose are reported as unavailable with the reason. |
| Recommendation engine | `makeai/hardware/recommend.py`, `benchmark.py` | Complexity 1/4/16/64/256 → architecture, context, precision, batch/accumulation, optimizer, LR, warmup, data requirement, VRAM/RAM/duration estimates. Budgets on the VRAM that is actually free and uses measured TFLOPS. AUTO or CUSTOM. |
| Model | `makeai/model/` | Decoder-only transformer: RMSNorm/LayerNorm, SwiGLU/GeGLU/GELU/ReLU/SiLU, RoPE (+linear/NTK scaling)/learned/ALiBi/none, MHA/GQA/MQA, sliding window, factorised embeddings, weight tying, gradient checkpointing. Layer names match Hugging Face Llama/Qwen2, so those weights load natively. |
| Datasets | `makeai/data/datasets.py` | JSON, JSONL, TXT, CSV, Parquet, folders; chat/instruction formats; cleaning, filtering, exact + MinHash dedup, shuffle, train/val/test split, packing, min/max length, token statistics. Chat data trains only on assistant replies. |
| Tokenizers | `makeai/data/tokenizer.py` | Byte-level BPE, WordPiece, Unigram, SentencePiece, custom (HF `tokenizer.json`); special tokens; Jinja chat templates. |
| Training | `makeai/train/` | Separate worker process per run. From scratch, full, LoRA, QLoRA (NF4/FP4, INT8 where supported), adapters. FP32/TF32/FP16/BF16. Adam, AdamW, AdamW 8-bit, Adafactor, Lion, SGD. Constant, linear, cosine, cosine with restarts, polynomial, one-cycle. Crash-safe checkpoints (auto, best, manual, pause, kill, final, cleanup, backup, resume). |
| Kill Switch | `makeai/train/runs.py` | Graceful: stops between micro-batches, saves the latest safe checkpoint, frees GPU memory, marks the run TERMINATED BY USER. Force/instant: terminates the process tree and marks the run interrupted. |
| Performance score | `makeai/train/health.py` | 1–10 score from measured GPU utilization, throughput versus measured peak, VRAM use, thermals/throttling, loss stability, data-loading wait and RAM pressure, with the main reason and a recommendation. It does not rate model quality. |
| Inference | `makeai/inference.py`, `model/generate.py` | Native engine with KV cache, llama.cpp for GGUF, ONNX Runtime for ONNX. Temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, max tokens, seed, stop sequences, system prompt, streaming, optional INT8/NF4/FP4 loading. |
| Import / export | `makeai/io/` | Import `.makeai`, `.safetensors` (+HF folders), `.gguf`, `.pt/.pth` (safe `weights_only` loading), `.onnx`. Export package, safetensors, HF folder, GGUF, PyTorch, ONNX (checked against PyTorch), PEFT LoRA adapter, tokenizer, config. TensorRT is reported as unavailable unless it is installed. |
| Attribution | `makeai/registry.py` | `makeai://user/name/version` IDs. Creator and original creator are written into every format's metadata and can never be replaced; importers are appended to `imported_by`. |
| Sharing / Discover | `makeai/sharing.py` | Private, link or public visibility; public pages; package downloads with counts; Discover merges this instance with other MakeAI hubs you add. |
| Projects | `makeai/projects.py` | Folders for src, models, datasets, tokenizers, configs, evaluation, tests and docs, with an editor and links to AIs, datasets and runs. |
| Dev agent (optional) | `makeai/devagent/` | Claude Code CLI: plan (read-only) → approve / edit / cancel → implement and test. Delete the folder and nothing else changes. |

## Claude Mode

Click **ClaudeMode** in the top bar: MakeAI turns white/orange and shows, live and read-only, what Claude does in the app - the activity feed on the left, the page Claude works on (for example its training dashboard) on the right. You cannot edit anything there, but you can write to Claude; it receives your messages while it works.

Claude connects through a real MCP server (`makeai/claudemode/mcp_server.py`, standard library only). Link it to Claude Code once (Claude Mode → Connect Claude → *Link automatically*, or):

```bash
claude mcp add --scope user makeai -e MAKEAI_URL=http://127.0.0.1:7860 -- <path>\.venv\Scripts\python.exe <path>\makeai\claudemode\mcp_server.py
```

Then either paste the prompt from *Connect Claude* into any Claude Code session (terminal, desktop app, IDE), or press **Start Claude** in Claude Mode, which runs your installed Claude Code headless with only the MakeAI tools (needs `claude auth login` once). Claude gets tools to inspect hardware and data, create AIs, train, pause/resume/kill, evaluate, chat and export - no delete tools. Claude Mode is optional: without it (or without Claude) MakeAI works exactly the same.

## Tests

```bash
.venv\Scripts\python -m pytest -q
```

`requirements.json` maps every requirement to its acceptance tests; the **Requirements & Tests** page runs them and shows the results. The tests train real models on the Python standard library source, kill and resume real training processes while measuring GPU memory through NVML, run GGUF exports in llama.cpp, load exports in Hugging Face `transformers` and PEFT, and check the runtime with the development agent removed.

## Known limits

- TensorRT export needs NVIDIA TensorRT, which is not bundled; build an engine from the ONNX export with `trtexec` instead.
- On Windows, CPU temperature is only available with administrator rights (ACPI) or while LibreHardwareMonitor/OpenHardwareMonitor is running.
- INT8 (LLM.int8) base quantization in bitsandbytes needs layer widths divisible by 32; MakeAI explains this instead of failing mid-run.
- The GGUF runtime in the default `llama-cpp-python` wheel is CPU-only; the MakeAI engine uses the GPU.

## License

MakeAI is **not open source**. See [LICENSE](LICENSE). In short: you may use MakeAI for free, and the AIs you create with it are yours to use, share, publish, sell and advertise. You may also promote MakeAI itself. You may **not** modify MakeAI or republish/redistribute it (original or modified) without written permission.
