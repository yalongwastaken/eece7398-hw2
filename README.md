# EECE7398 Project 2 — Voice Interaction System

End-to-end voice Q&A system built with Whisper (ASR), Qwen2.5-7B (LLM), and Kokoro (TTS).
Benchmarked on Northeastern's Explorer HPC cluster, deployed for inference on macOS (Apple Silicon).

## Pipeline

```
mic input → Whisper (ASR) → Qwen2.5-7B (LLM) → Kokoro (TTS) → audio output
```

## Project Structure

```
eece7398-hw2/
├── build.sh              # OS-aware llama.cpp build script (CUDA on Explorer, Metal on Mac)
├── requirements.txt      # Python dependencies
├── config.yaml           # model paths and settings
├── models/               # model weights (gitignored, contents only)
├── scripts/
│   └── download_models.sh  # downloads all model weights from HuggingFace
├── src/                  # ASR, LLM, TTS modules + pipeline
├── benchmark/            # timing and evaluation scripts
├── ui/                   # frontend (bonus)
└── report/               # report assets
```

## Requirements

- Python 3.12
- CMake
- Xcode Command Line Tools (`xcode-select --install`)
- CUDA toolkit (Explorer only)

## Setup

### macOS (Apple Silicon)

> Tested on macOS 26 (Tahoe) with Xcode Command Line Tools.

```bash
git clone <repo-url>
cd eece7398-hw2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
bash scripts/download_models.sh
```

### Explorer (HPC)

```bash
git clone <repo-url>
cd eece7398-hw2
module load cuda  # load appropriate CUDA module
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
bash scripts/download_models.sh
```

## Usage

```bash
source .venv/bin/activate

# run full pipeline
python src/pipeline.py

# benchmark individual components
python benchmark/benchmark.py --component llm
python benchmark/benchmark.py --component asr
python benchmark/benchmark.py --component tts
```

## Models

| Component | Model | Source | Format |
|-----------|-------|--------|--------|
| ASR | Whisper small | openai-whisper | - |
| LLM | Qwen2.5-7B-Instruct | bartowski/Qwen2.5-7B-Instruct-GGUF | Q4_K_M GGUF |
| TTS | Kokoro | kokoro pip package | - |

## Notes

- `llama.cpp/` is cloned and built locally by `build.sh` (gitignored)
- Model weights go in `models/` (gitignored)
- `build.sh` auto-detects OS and builds with Metal (Mac) or CUDA (Linux)

## Authors

Anthony — Northeastern University, EECE7398