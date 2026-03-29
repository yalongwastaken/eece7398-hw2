# test/test_llm.py
# quick sanity check for Qwen2.5-7B via llama.cpp
# usage: python test/test_llm.py

import subprocess
import time
import yaml
import os

def main():
    # load config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["llm"]["model_path"]
    binary = os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "build", "bin", "llama-cli")

    print(f"[test_llm] model:  {model_path}")
    print(f"[test_llm] binary: {binary}")
    print(f"[test_llm] prompt: What is the capital of France?")
    print("-" * 40)

    cmd = [
        binary,
        "-m", model_path,
        "-p", "What is the capital of France?",
        "-n", "64",
        "--no-display-prompt",
        "--single-turn",
        "-ngl", str(cfg["llm"]["n_gpu_layers"]),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(result.stdout.strip())
    print("-" * 40)
    print(f"[test_llm] wall time: {elapsed:.2f}s")

    if result.returncode != 0:
        print("[test_llm] FAILED")
        print(result.stderr)
    else:
        print("[test_llm] PASSED")

if __name__ == "__main__":
    main()