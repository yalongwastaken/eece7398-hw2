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
├── build.sh              # OS-aware llama.cpp build script
├── requirements.txt      # Python dependencies
├── config.yaml           # model paths and settings
├── models/               # model weights (gitignored)
├── scripts/              # SBATCH job scripts
├── src/                  # ASR, LLM, TTS modules + pipeline
├── benchmark/            # timing and evaluation scripts
├── ui/                   # frontend (bonus)
└── report/               # report assets
```

## Setup

### Explorer (HPC)

```bash
git clone https://github.com/yalongwastaken/eece7398-hw2.git
cd eece7398-hw2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
```

### macOS (Apple Silicon)

```bash
git clone <repo-url>
cd eece7398-hw2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
```

Download Qwen2.5-7B Q4_K_M from HuggingFace into `models/`.

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

| Component | Model | Format |
|-----------|-------|--------|
| ASR | Whisper small | openai-whisper |
| LLM | Qwen2.5-7B-Instruct | Q4_K_M GGUF |
| TTS | Kokoro | - |

## Authors

Anthony — Northeastern University, EECE7398