# @file    test_tts.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Sanity check for Kokoro TTS. Synthesizes a test sentence and saves
#          the output to test/output_tts.wav. Reports RTF and wall time.
# @usage   python test/test_tts.py

# imports
import os
import time
import warnings
import logging
import yaml
import numpy as np
import soundfile as sf
from kokoro import KPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

TEST_TEXT = "The capital of France is Paris. This is a test of the Kokoro text to speech system."
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output_tts.wav")

def main() -> None:
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

    # synthesize and collect audio chunks
    print("[test_tts] synthesizing...")
    t0 = time.time()
    chunks = []
    for _, _, audio in pipeline(TEST_TEXT, voice=voice, speed=1.0):
        chunks.append(audio)
    elapsed = time.time() - t0

    audio = np.concatenate(chunks)
    sf.write(OUTPUT_PATH, audio, sample_rate)

    duration = len(audio) / sample_rate
    rtf = elapsed / duration

    print("-" * 40)
    print(f"[test_tts] output:    {OUTPUT_PATH}")
    print(f"[test_tts] duration:  {duration:.2f}s of audio")
    print(f"[test_tts] wall time: {elapsed:.2f}s")
    print(f"[test_tts] RTF:       {rtf:.3f}")
    print("-" * 40)
    print("[test_tts] PASSED")

if __name__ == "__main__":
    main()