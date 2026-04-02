# @file    asr.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Whisper ASR wrapper. Loads the Whisper model once at init and
#          exposes a single transcribe() method that accepts a 16kHz float32
#          numpy array and returns the transcript text and elapsed time.

# imports
import time
import numpy as np
import whisper

class ASR:
    def __init__(self, cfg: dict) -> None:
        # load whisper model and store inference settings
        self.model = whisper.load_model(cfg["model"])
        self.language = cfg["language"]
        self.fp16 = cfg.get("fp16", False)

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        # audio must be float32 at 16kHz
        t0 = time.time()
        result = self.model.transcribe(audio, language=self.language, fp16=self.fp16)
        elapsed = time.time() - t0
        return result["text"].strip(), elapsed