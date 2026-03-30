# src/asr.py
# Whisper ASR wrapper

import whisper
import time

class ASR:
    def __init__(self, cfg):
        self.model    = whisper.load_model(cfg["model"])
        self.language = cfg["language"]
        self.fp16     = cfg.get("fp16", False)

    def transcribe(self, audio):
        # audio: numpy float32 array at 16kHz
        t0 = time.time()
        result = self.model.transcribe(audio, language=self.language, fp16=self.fp16)
        elapsed = time.time() - t0
        return result["text"].strip(), elapsed