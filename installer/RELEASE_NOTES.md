**MakeAI by Convergent** — create, train and run your own AI models on your own computer. No cloud, no account.

### Download & install (Windows 10/11, 64-bit)

1. Download **MakeAI-Setup.exe** below and run it.
2. Accept the license, choose a folder, click **Install**.
3. Setup downloads Python 3.10 and PyTorch (about 2.5 GB with an NVIDIA GPU, about 200 MB without) and puts **MakeAI** on your desktop and in the Start menu.

Windows may show "Windows protected your PC" because the installer is not code-signed: click **More info → Run anyway**. You can check the file with the SHA-256 in `MakeAI-Setup.exe.sha256`.

**Requirements:** Windows 10/11 x64, ~6 GB free disk space, internet during setup. An NVIDIA GPU (CUDA) is strongly recommended for training; without one MakeAI trains on the CPU. MakeAI opens in its own window using Google Chrome or Microsoft Edge (otherwise in your default browser).

### What's inside

- Hardware scan and live GPU/CPU/RAM monitoring, hardware-aware recommendations (complexity 1 · 4 · 16 · 64 · 256)
- Create AIs from scratch or fine-tune (full, LoRA, QLoRA, adapters), datasets and tokenizers
- Live training view with loss curve, performance score, pause/resume and a Kill Switch that saves a safe checkpoint
- Chat with your AIs locally; import/export safetensors, GGUF, ONNX, PyTorch and MakeAI packages; creator attribution that survives every export
- **Claude Mode** (optional): link MakeAI to Claude Code and watch Claude create and train AIs for you

Your AIs and data are stored in `%USERPROFILE%\MakeAI` and are kept when you update or uninstall ("Uninstall MakeAI.cmd" in the install folder).

MakeAI is not open source — see [LICENSE](https://github.com/eronbubi/MakeAI-/blob/main/LICENSE): free to use; the AIs you make are yours to share, publish and sell; the app itself may not be modified or redistributed.
