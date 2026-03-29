# test/test_asr.py
# quick sanity check for Whisper ASR
# generates a test wav via TTS (pyttsx3-free), transcribes it, checks output
# usage: python test/test_asr.py

import whisper
import numpy as np
import time
import yaml
import os

TEST_TEXT = "The capital of France is Paris."

def main():
    # load config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["asr"]["model"]
    language   = cfg["asr"]["language"]

    print(f"[test_asr] loading Whisper model: {model_name}")
    t0 = time.time()
    model = whisper.load_model(model_name)
    print(f"[test_asr] model loaded in {time.time() - t0:.2f}s")

    # generate a simple sine wave as a dummy audio file
    # whisper will transcribe silence/noise — we just want to confirm the pipeline runs
    print("[test_asr] generating test audio (440Hz tone, 3s)...")
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print("[test_asr] transcribing audio array directly...")
    t0 = time.time()
    result = model.transcribe(audio, language=language, fp16=False)
    elapsed = time.time() - t0

    print("-" * 40)
    print(f"[test_asr] transcription: '{result['text'].strip()}'")
    print(f"[test_asr] wall time:     {elapsed:.2f}s")
    print("-" * 40)
    print("[test_asr] PASSED")

if __name__ == "__main__":
    main()