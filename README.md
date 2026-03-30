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
├── build.sh                  # OS-aware llama.cpp build script (CUDA on Explorer, Metal on Mac)
├── requirements.txt          # Python dependencies
├── config.yaml               # model paths and settings
├── models/                   # model weights (gitignored, contents only)
├── scripts/
│   └── download_models.sh    # downloads all model weights from HuggingFace
├── src/
│   ├── asr.py                # Whisper ASR wrapper
│   ├── llm.py                # Qwen2.5-7B wrapper via llama-server
│   ├── tts.py                # Kokoro TTS wrapper
│   └── pipeline.py           # end-to-end voice Q&A pipeline
├── test/
│   ├── test_llm.py           # LLM sanity check
│   ├── test_asr.py           # ASR sanity check
│   └── test_tts.py           # TTS sanity check
├── benchmark/
│   └── benchmark.py          # per-component benchmarks
├── ui/                       # frontend (bonus)
└── report/                   # report assets
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

> **SSL fix (python.org installs only):** If you get `SSL: CERTIFICATE_VERIFY_FAILED` when downloading models, run:
> ```bash
> /Applications/Python\ 3.12/Install\ Certificates.command
> ```

## Usage

### Run the pipeline

```bash
source .venv/bin/activate

# start the LLM server (keep running in a separate terminal)
./llama.cpp/build/bin/llama-server \
    -m models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    -ngl 35 -c 2048 -t 8 \
    --host 127.0.0.1 --port 8080 --log-disable

# run the pipeline
python src/pipeline.py
```

Press Enter to start recording, speak your question, press Enter to stop.

### Run tests

```bash
python test/test_llm.py
python test/test_asr.py
python test/test_tts.py
```

### Run benchmarks

```bash
# start llama-server first (see above), then:
python benchmark/benchmark.py --component asr
python benchmark/benchmark.py --component llm
python benchmark/benchmark.py --component tts
python benchmark/benchmark.py --component all
```

## Benchmark Results (Apple M3 Pro, macOS 26, Metal backend)

### ASR — Whisper small

| Metric | Value |
|--------|-------|
| WER | 0.00 |
| Mean RTF | 0.166 |
| Throughput | 8.8x real-time |
| Mean latency | 0.59s |

### LLM — Qwen2.5-7B-Instruct Q4_K_M

| Metric | Value |
|--------|-------|
| Generation speed | ~28 tok/s |
| Mean latency | 1.40s |
| Prompt processing | 19–120 tok/s |

### TTS — Kokoro (af_heart)

| Metric | Value |
|--------|-------|
| Mean RTF | 0.140 |
| Throughput | 7.1x real-time |
| Mean synthesis time | 0.85s |

## Models

| Component | Model | Source | Format |
|-----------|-------|--------|--------|
| ASR | Whisper small | openai-whisper | - |
| LLM | Qwen2.5-7B-Instruct | bartowski/Qwen2.5-7B-Instruct-GGUF | Q4_K_M GGUF |
| TTS | Kokoro | hexgrad/Kokoro-82M | - |

## Notes

- `llama.cpp/` is cloned and built locally by `build.sh` (gitignored)
- Model weights go in `models/` (gitignored)
- `build.sh` auto-detects OS and builds with Metal (Mac) or CUDA (Linux)
- Whisper runs in FP32 mode on CPU (FP16 requires CUDA)
- LLM inference runs via `llama-server` HTTP API on localhost:8080