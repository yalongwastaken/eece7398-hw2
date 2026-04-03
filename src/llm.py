# @file    llm.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Qwen2.5-7B-Instruct wrapper via llama-server HTTP API.
#          Spawns llama-server as a background process on init, waits for it
#          to become healthy, then exposes a generate() method that sends
#          requests to the OpenAI-compatible /v1/chat/completions endpoint.
#          Call shutdown() to terminate the server process on exit.

# imports
import os
import time
import signal
import logging
import subprocess
import requests

# logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

SERVER_URL = "http://localhost:8080"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant embedded in a real-time voice interaction system. "
    "Your responses will be converted to speech, so follow these rules:\n"
    "- Answer in 1-2 sentences only — never more.\n"
    "- Use plain spoken language. Avoid bullet points, markdown, or special characters.\n"
    "- Spell out abbreviations (say 'I squared C' not 'I2C', 'real-time OS' not 'RTOS').\n"
    "- Be direct and confident. Do not hedge or over-qualify.\n"
    "- If a question is technical, give a clear one-sentence definition followed by one practical example."
)

class LLM:
    def __init__(self, cfg: dict) -> None:
        # setup
        root = os.path.join(os.path.dirname(__file__), "..")
        binary = os.path.join(root, "llama.cpp", "build", "bin", "llama-server")
        model = os.path.join(root, cfg["model_path"])

        # build llama-server launch command
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

        # launch server as background process
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # poll health endpoint until server is ready (up to 30s)
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

    def generate(self, user_text: str) -> tuple[str, float]:
        # format as OpenAI-compatible chat completion request
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

    def shutdown(self) -> None:
        # send SIGTERM to gracefully stop the server process
        if self.proc:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait()