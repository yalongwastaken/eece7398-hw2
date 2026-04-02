# @file    pipeline.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   End-to-end voice Q&A pipeline. Records mic input on Enter key,
#          transcribes with Whisper, generates a response with Qwen2.5-7B,
#          synthesizes speech with Kokoro, and plays it back. Loops until
#          Ctrl+C.
# @usage   python src/pipeline.py

# imports
import os
import sys
import time
import yaml
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.asr import ASR
from src.llm import LLM
from src.tts import TTS

SAMPLE_RATE = 16000

def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def record_until_enter() -> np.ndarray:
    # stream mic audio into chunks until user presses Enter
    print("[pipeline] recording... press Enter to stop")
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()

    return np.concatenate(chunks, axis=0).squeeze()


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    sd.play(audio, sample_rate)
    sd.wait()


def main() -> None:
    cfg = load_config()

    # load all models at startup
    print("[pipeline] loading models...")
    t0 = time.time()
    asr = ASR(cfg["asr"])
    llm = LLM(cfg["llm"])
    tts = TTS(cfg["tts"])
    print(f"[pipeline] models loaded in {time.time() - t0:.2f}s")
    print("[pipeline] ready. press Ctrl+C to quit.\n")

    while True:
        try:
            input("[pipeline] press Enter to start recording...")
            audio = record_until_enter()

            # ASR
            print("[pipeline] transcribing...")
            text, asr_time = asr.transcribe(audio)
            print(f"[pipeline] you said: '{text}' ({asr_time:.2f}s)")

            if not text:
                print("[pipeline] no speech detected, try again.")
                continue

            # LLM
            print("[pipeline] generating response...")
            response, llm_time = llm.generate(text)
            print(f"[pipeline] response: '{response}' ({llm_time:.2f}s)")

            # TTS
            print("[pipeline] synthesizing speech...")
            speech, tts_time = tts.synthesize(response)
            print(f"[pipeline] playing ({tts_time:.2f}s synthesis)...")
            play_audio(speech, tts.sample_rate)

            total = asr_time + llm_time + tts_time
            print(f"[pipeline] total latency: {total:.2f}s (ASR {asr_time:.2f} + LLM {llm_time:.2f} + TTS {tts_time:.2f})\n")

        except KeyboardInterrupt:
            print("\n[pipeline] exiting.")
            llm.shutdown()
            break


if __name__ == "__main__":
    main()