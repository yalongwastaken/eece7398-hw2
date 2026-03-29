# test/test_tts.py
# quick sanity check for Kokoro TTS
# generates speech from text and saves to test/output_tts.wav
# usage: python test/test_tts.py

import time
import yaml
import os
import warnings
import logging
import numpy as np
import soundfile as sf
from kokoro import KPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

TEST_TEXT = "The capital of France is Paris. This is a test of the Kokoro text to speech system."
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output_tts.wav")

def main():
    # load config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    voice = cfg["tts"]["voice"]
    sample_rate = cfg["tts"]["sample_rate"]

    print(f"[test_tts] voice:       {voice}")
    print(f"[test_tts] sample rate: {sample_rate}")
    print(f"[test_tts] text:        {TEST_TEXT}")
    print("-" * 40)

    print("[test_tts] initializing Kokoro pipeline...")
    t0 = time.time()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    print(f"[test_tts] pipeline ready in {time.time() - t0:.2f}s")

    print("[test_tts] synthesizing...")
    t0 = time.time()
    audio_chunks = []
    generator = pipeline(TEST_TEXT, voice=voice, speed=1.0)
    for _, _, audio in generator:
        audio_chunks.append(audio)
    elapsed = time.time() - t0

    audio = np.concatenate(audio_chunks)
    sf.write(OUTPUT_PATH, audio, sample_rate)

    duration = len(audio) / sample_rate
    rtf = elapsed / duration  # real-time factor

    print("-" * 40)
    print(f"[test_tts] output:    {OUTPUT_PATH}")
    print(f"[test_tts] duration:  {duration:.2f}s of audio")
    print(f"[test_tts] wall time: {elapsed:.2f}s")
    print(f"[test_tts] RTF:       {rtf:.3f} (lower is faster than real-time)")
    print("-" * 40)
    print("[test_tts] PASSED")

if __name__ == "__main__":
    main()