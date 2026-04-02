# @file    test_asr.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Sanity check for Whisper ASR. Generates a synthetic 440Hz tone
#          and transcribes it directly as a numpy array to verify the pipeline
#          runs end-to-end without errors.
# @usage   python test/test_asr.py

# imports
import os
import time
import yaml
import numpy as np
import whisper

def main() -> None:
    # load config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["asr"]["model"]
    language = cfg["asr"]["language"]

    print(f"[test_asr] loading Whisper model: {model_name}")
    t0 = time.time()
    model = whisper.load_model(model_name)
    print(f"[test_asr] model loaded in {time.time() - t0:.2f}s")

    # generate a 440Hz sine tone — empty transcription is expected, pipeline just needs to run
    print("[test_asr] generating test audio (440Hz, 3s)...")
    sr = 16000
    t = np.linspace(0, 3, int(sr * 3))
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print("[test_asr] transcribing...")
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