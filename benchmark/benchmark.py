# benchmark/benchmark.py
# comprehensive benchmark for ASR, LLM, and TTS components
# usage: python benchmark/benchmark.py --component [asr|llm|tts|all]

import os
import sys
import time
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

N_RUNS = 5  # number of runs to average per sample

# --- test data ---

ASR_SAMPLES = [
    ("Hello.",                                                                                     1.0),
    ("The quick brown fox jumps over the lazy dog.",                                               3.0),
    ("Debouncing is a technique used to filter out rapid repeated signals from a button.",         5.0),
    ("Embedded systems require careful management of hardware resources and real-time constraints.",6.0),
    ("An interrupt service routine is a callback function executed by the processor in response "
     "to a hardware or software interrupt signal.",                                                8.0),
]

LLM_PROMPTS = [
    ("What is two plus two?",                                                                      "short"),
    ("What is debouncing?",                                                                        "short"),
    ("Explain the difference between RISC and CISC architectures.",                               "medium"),
    ("How does I2C communication work?",                                                           "medium"),
    ("Describe the role of a real-time operating system in embedded systems "
     "and give two examples of commonly used RTOSes.",                                            "long"),
]

TTS_TEXTS = [
    ("Hello.",                                                                                     "very_short"),
    ("How can I assist you today?",                                                                "short"),
    ("Debouncing filters out rapid repeated signals from a switching device.",                     "medium"),
    ("Real-time operating systems prioritize tasks based on strict timing deadlines, "
     "ensuring deterministic behavior in safety-critical applications.",                           "long"),
    ("The I squared C protocol uses two wires, a serial data line and a serial clock line, "
     "to enable communication between a master device and one or more slave devices "
     "on the same bus.",                                                                           "very_long"),
]

TTS_VOICES = ["af_heart", "af_bella", "am_adam"]

def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def print_stats(label, values):
    arr = np.array(values)
    print(f"  {label}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
          f"min={arr.min():.3f}  max={arr.max():.3f}")

# --- ASR benchmark ---

def benchmark_asr(cfg):
    import whisper
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    from kokoro import KPipeline

    print("\n" + "="*60)
    print("ASR BENCHMARK — Whisper")
    print("="*60)
    print(f"  model: {cfg['model']} (avg over {N_RUNS} runs)")

    t0 = time.time()
    model = whisper.load_model(cfg["model"])
    print(f"  model load time: {time.time() - t0:.2f}s")

    print("  generating test audio via Kokoro TTS...")
    tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    sr = 16000

    def tts_to_audio(text):
        chunks = []
        for _, _, audio in tts(text, voice="af_heart", speed=1.0):
            chunks.append(audio)
        audio = np.concatenate(chunks)
        # resample 24kHz -> 16kHz
        n = int(len(audio) * sr / 24000)
        return np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio).astype(np.float32)

    times, rtfs, wers, audio_durations = [], [], [], []
    print(f"\n  {'#':<4} {'dur':>6} {'time':>7} {'RTF':>7} {'WER':>6}  transcript")
    print(f"  {'-'*75}")

    for i, (text, _) in enumerate(ASR_SAMPLES):
        audio = tts_to_audio(text)
        duration = len(audio) / sr

        run_times = []
        for _ in range(N_RUNS):
            t0 = time.time()
            result = model.transcribe(audio, language=cfg["language"], fp16=cfg.get("fp16", False))
            run_times.append(time.time() - t0)

        elapsed = np.mean(run_times)
        rtf = elapsed / duration
        transcript = result["text"].strip()

        ref = text.lower().split()
        hyp = transcript.lower().split()
        edits = sum(1 for r, h in zip(ref, hyp) if r != h) + abs(len(ref) - len(hyp))
        wer = edits / max(len(ref), 1)

        times.append(elapsed)
        rtfs.append(rtf)
        wers.append(wer)
        audio_durations.append(duration)

        print(f"  [{i+1}] {duration:>5.1f}s  {elapsed:>6.2f}s  {rtf:>6.3f}  {wer:>5.2f}  '{transcript[:45]}'")

    print(f"\n  --- summary ---")
    print_stats("latency (s)", times)
    print_stats("RTF      ", rtfs)
    print_stats("WER      ", wers)
    print(f"  throughput: {sum(audio_durations)/sum(times):.2f}x real-time")

# --- LLM benchmark ---

def benchmark_llm(cfg):
    import requests

    print("\n" + "="*60)
    print("LLM BENCHMARK — Qwen2.5-7B-Instruct (llama-server)")
    print("="*60)
    SERVER_URL = "http://localhost:8080"
    SYSTEM = "You are a helpful assistant. Answer in 1-2 sentences only."
    MAX_TOKENS = 64  # keep short for benchmarking latency

    print(f"  NOTE: llama-server must be running on localhost:8080")
    print(f"  max_tokens: {MAX_TOKENS} (avg over {N_RUNS} runs)\n")

    times, gen_tps, gen_tokens = [], [], []
    print(f"  {'#':<4} {'type':<8} {'time':>7} {'p_tps':>8} {'g_tps':>8} {'tok':>5}  response")
    print(f"  {'-'*75}")

    for i, (prompt, length) in enumerate(LLM_PROMPTS):
        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }

        run_times = []
        for _ in range(N_RUNS):
            t0 = time.time()
            resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
            run_times.append(time.time() - t0)

        elapsed = np.mean(run_times)
        data = resp.json()
        if "choices" not in data:
            print(f"  [{i+1}] ERROR: unexpected response: {data}")
            continue
        response = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        p_tokens = usage.get("prompt_tokens", 0)
        g_tokens = usage.get("completion_tokens", 0)
        p_tps = p_tokens / elapsed if elapsed > 0 else 0
        g_tps = g_tokens / elapsed if elapsed > 0 else 0

        times.append(elapsed)
        gen_tps.append(g_tps)
        gen_tokens.append(g_tokens)

        print(f"  [{i+1}] {length:<8} {elapsed:>6.2f}s  {p_tps:>7.1f}  {g_tps:>7.1f}  {g_tokens:>4}  '{response[:45]}...'")

    print(f"\n  --- summary ---")
    print_stats("latency (s)     ", times)
    print_stats("gen tokens/req  ", gen_tokens)
    print_stats("approx gen tps  ", gen_tps)

# --- TTS benchmark ---

def benchmark_tts(cfg):
    import warnings
    import logging
    import soundfile as sf
    warnings.filterwarnings("ignore")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    from kokoro import KPipeline

    print("\n" + "="*60)
    print("TTS BENCHMARK — Kokoro")
    print("="*60)

    t0 = time.time()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    print(f"  model load time: {time.time() - t0:.2f}s")
    print(f"  voices tested: {TTS_VOICES} (avg over {N_RUNS} runs)\n")

    sr = cfg["sample_rate"]
    out_dir = os.path.dirname(__file__)

    for voice in TTS_VOICES:
        print(f"  voice: {voice}")
        print(f"  {'#':<4} {'label':<12} {'dur':>6} {'time':>7} {'RTF':>7}")
        print(f"  {'-'*45}")

        times, rtfs = [], []
        for i, (text, label) in enumerate(TTS_TEXTS):
            run_times = []
            audio = None
            for _ in range(N_RUNS):
                t0 = time.time()
                chunks = []
                for _, _, a in pipeline(text, voice=voice, speed=1.0):
                    chunks.append(a)
                run_times.append(time.time() - t0)
                audio = np.concatenate(chunks)

            elapsed = np.mean(run_times)
            duration = len(audio) / sr
            rtf = elapsed / duration
            times.append(elapsed)
            rtfs.append(rtf)

            out = os.path.join(out_dir, f"tts_{voice}_{label}.wav")
            sf.write(out, audio, sr)
            print(f"  [{i+1}] {label:<12} {duration:>5.2f}s  {elapsed:>6.2f}s  {rtf:>6.3f}")

        print(f"  avg RTF: {np.mean(rtfs):.3f}  avg time: {np.mean(times):.2f}s\n")

# --- main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["asr", "llm", "tts", "all"], default="all")
    args = parser.parse_args()
    cfg = load_config()

    if args.component in ("asr", "all"):
        benchmark_asr(cfg["asr"])
    if args.component in ("llm", "all"):
        benchmark_llm(cfg["llm"])
    if args.component in ("tts", "all"):
        benchmark_tts(cfg["tts"])

    print("\n" + "="*60)
    print("benchmark complete.")
    print("="*60)

if __name__ == "__main__":
    main()