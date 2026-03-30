# src/tts.py
# Kokoro TTS wrapper

import warnings
import logging
import numpy as np
import time

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from kokoro import KPipeline

class TTS:
    def __init__(self, cfg):
        self.voice       = cfg["voice"]
        self.sample_rate = cfg["sample_rate"]
        self.pipeline    = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def synthesize(self, text):
        if not text:
            return np.zeros(0, dtype=np.float32), 0.0
        # returns (audio np.float32 array, elapsed seconds)
        t0 = time.time()
        chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=1.0):
            chunks.append(audio)
        elapsed = time.time() - t0
        return np.concatenate(chunks), elapsed