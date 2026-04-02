# EECE7398 Project 2 — Voice Interaction System

- **Author:** Anthony Yalong
- **NUID:** 002156860
- **Course:** EECE7398 — Deep Learning Embedded Systems
- End-to-end voice Q&A system built with Whisper (ASR), Qwen2.5-7B (LLM), and Kokoro (TTS). Deployed on macOS Apple Silicon (Metal backend).

## Pipeline

```
mic input → Whisper (ASR) → Qwen2.5-7B (LLM) → Kokoro (TTS) → audio output
```

## Project Structure

```
eece7398-hw2/
├── build.sh                 # builds llama.cpp with Metal backend
├── requirements.txt         # Python dependencies
├── config.yaml              # model paths and inference settings
├── models/                  # model weights (gitignored, contents only)
├── scripts/
│   └── download_models.sh   # downloads Qwen2.5-7B GGUF from HuggingFace
├── src/
│   ├── asr.py               # Whisper ASR wrapper
│   ├── llm.py               # Qwen2.5-7B wrapper via llama-server
│   ├── tts.py               # Kokoro TTS wrapper
│   └── pipeline.py          # end-to-end voice Q&A pipeline
├── test/
│   ├── test_llm.py          # LLM sanity check
│   ├── test_asr.py          # ASR sanity check
│   ├── test_tts.py          # TTS sanity check
│   ├── test_voice_cloning.py          # voice cloning evaluation
│   └── test_prompt_engineering.py    # prompt engineering comparison
├── benchmark/
│   └── benchmark.py         # per-component benchmarks (ASR, LLM, TTS)
├── ui/
│   └── app.py               # Gradio voice Q&A frontend
└── report/                  # report assets
```

## Requirements

- Python 3.12
- CMake
- Xcode Command Line Tools (`xcode-select --install`)

## Setup

> Tested on macOS 26 (Tahoe), Apple M3 Pro, Xcode Command Line Tools.

```bash
git clone <repo-url>
cd eece7398-hw2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
bash scripts/download_models.sh
```

> **SSL fix (python.org installs only):** If you get `SSL: CERTIFICATE_VERIFY_FAILED`, run:
> ```bash
> /Applications/Python\ 3.12/Install\ Certificates.command
> ```

## Usage

### Run the pipeline

```bash
source .venv/bin/activate

# terminal 1 — start LLM server
./llama.cpp/build/bin/llama-server \
    -m models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    -ngl 35 -c 2048 -t 8 \
    --host 127.0.0.1 --port 8080 --log-disable

# terminal 2 — run pipeline
python src/pipeline.py
```

Press Enter to start recording, speak, press Enter to stop.

### Run the UI

```bash
source .venv/bin/activate

# terminal 1 — start LLM server
./llama.cpp/build/bin/llama-server \
    -m models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    -ngl 35 -c 2048 -t 8 \
    --host 127.0.0.1 --port 8080 --log-disable

# terminal 2 — run app
python ui/app.py
```

Open `http://127.0.0.1:7860` in your browser.

### Run tests

```bash
python test/test_llm.py
python test/test_asr.py
python test/test_tts.py
python test/test_voice_cloning.py
python test/test_prompt_engineering.py  # requires llama-server running
```

### Run benchmarks

```bash
# start llama-server first (see above), then:
python benchmark/benchmark.py --component all
```

Or run individual components:

```bash
python benchmark/benchmark.py --component asr
python benchmark/benchmark.py --component llm
python benchmark/benchmark.py --component tts
```

## Benchmark Results (Apple M3 Pro, macOS 26, Metal)

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
| TTS | Kokoro-82M | hexgrad/Kokoro-82M | - |

## Notes

- `llama.cpp/` is cloned and built by `build.sh` (gitignored)
- Model weights go in `models/` (gitignored)
- Whisper runs in FP32 on CPU (FP16 requires CUDA)
- LLM inference runs via `llama-server` HTTP API on localhost:8080
- Whisper and Kokoro weights download automatically on first run