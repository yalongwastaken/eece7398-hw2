# @file    tts.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Kokoro TTS wrapper. Loads the Kokoro-82M pipeline once at init
#          and exposes a synthesize() method that converts text to a float32
#          audio array. Returns empty audio for empty input.

# imports
import time
import warnings
import logging
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from kokoro import KPipeline

class TTS:
    def __init__(self, cfg: dict) -> None:
        # load Kokoro pipeline with American English
        self.voice = cfg["voice"]
        self.sample_rate = cfg["sample_rate"]
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        # return silent audio for empty input
        if not text:
            return np.zeros(0, dtype=np.float32), 0.0

        t0 = time.time()
        chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=1.0):
            chunks.append(audio)
        elapsed = time.time() - t0
        return np.concatenate(chunks), elapsed