# src/llm.py
# Qwen2.5-7B-Instruct wrapper via llama-server HTTP API

import subprocess
import logging
import requests
import signal
import time
import os

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

SERVER_URL = "http://localhost:8080"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Answer in 1-2 sentences only. Be concise and direct."
)

class LLM:
    def __init__(self, cfg):
        root = os.path.join(os.path.dirname(__file__), "..")
        binary = os.path.join(root, "llama.cpp", "build", "bin", "llama-server")
        model  = os.path.join(root, cfg["model_path"])

        cmd = [
            binary,
            "-m", model,
            "-ngl", str(cfg["n_gpu_layers"]),
            "-c", str(cfg["n_ctx"]),
            "-t", str(cfg["n_threads"]),
            "--host", "127.0.0.1",
            "--port", "8080",
            "--log-disable",
        ]

        self.max_tokens = cfg["max_tokens"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # wait for server to be ready
        print("[llm] starting server...", end="", flush=True)
        for _ in range(30):
            try:
                requests.get(f"{SERVER_URL}/health", timeout=1)
                print(" ready.")
                break
            except Exception:
                time.sleep(1)
                print(".", end="", flush=True)
        else:
            raise RuntimeError("llama-server failed to start")

    def generate(self, user_text):
        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
        }

        t0 = time.time()
        resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        elapsed = time.time() - t0

        response = resp.json()["choices"][0]["message"]["content"].strip()
        return response, elapsed

    def shutdown(self):
        if self.proc:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait()