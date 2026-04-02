# @file    test_prompt_engineering.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Evaluates the effect of prompt engineering on LLM response quality.
#          Tests 3 system prompts across 5 embedded systems questions and saves
#          results to a JSON file for report analysis.
# @note    llama-server must be running on localhost:8080
# @usage   python test/test_prompt_engineering.py

# imports
import os
import sys
import time
import json
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SERVER_URL  = "http://localhost:8080"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "prompt_engineering_results.json")

# system prompts to compare
PROMPTS = {
    "baseline": None,  # no system prompt

    "constrained": (
        "You are a helpful voice assistant. "
        "Answer in 1-2 sentences only. Be concise and direct."
    ),

    "enhanced": (
        "You are a helpful voice assistant embedded in a real-time voice interaction system. "
        "Your responses will be converted to speech, so follow these rules:\n"
        "- Answer in 1-2 sentences only — never more.\n"
        "- Use plain spoken language. Avoid bullet points, markdown, or special characters.\n"
        "- Spell out abbreviations (say 'I squared C' not 'I2C', 'real-time OS' not 'RTOS').\n"
        "- Be direct and confident. Do not hedge or over-qualify.\n"
        "- If a question is technical, give a clear one-sentence definition followed by one practical example."
    ),
}

# test questions
QUESTIONS = [
    "What is debouncing in embedded systems?",
    "Explain I2C communication.",
    "What is a watchdog timer used for?",
    "What is the difference between microcontrollers and microprocessors?",
    "What does RTOS stand for and why is it used?",
]

def query(system_prompt: str | None, user_text: str, max_tokens: int = 128) -> tuple[str, float, int]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,  # low temp for consistent comparison
    }

    t0 = time.time()
    resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
    elapsed = time.time() - t0

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return text, elapsed, tokens


def main() -> None:
    print("\n" + "="*70)
    print("PROMPT ENGINEERING EVALUATION")
    print("="*70)
    print(f"  {len(PROMPTS)} prompts x {len(QUESTIONS)} questions\n")

    results = {}

    for prompt_name, system_prompt in PROMPTS.items():
        print(f"\n--- Prompt: {prompt_name} ---")
        if system_prompt:
            print(f"  system: \"{system_prompt[:80].replace(chr(10), ' ')}...\"")
        else:
            print(f"  system: (none)")

        results[prompt_name] = []
        total_tokens = []
        total_times = []

        for i, question in enumerate(QUESTIONS):
            response, elapsed, tokens = query(system_prompt, question)
            total_tokens.append(tokens)
            total_times.append(elapsed)

            results[prompt_name].append({
                "question": question,
                "response": response,
                "tokens": tokens,
                "latency": round(elapsed, 3),
            })

            print(f"\n  Q{i+1}: {question}")
            print(f"  A:  {response}")
            print(f"      [{tokens} tokens, {elapsed:.2f}s]")

        print(f"\n  avg tokens:  {sum(total_tokens)/len(total_tokens):.1f}")
        print(f"  avg latency: {sum(total_times)/len(total_times):.2f}s")

    # save full results to json
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"results saved to {OUTPUT_PATH}")

    # summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for pname, entries in results.items():
        avg_tok = sum(e["tokens"] for e in entries) / len(entries)
        avg_lat = sum(e["latency"] for e in entries) / len(entries)
        print(f"  {pname:<14} avg_tokens={avg_tok:5.1f}  avg_latency={avg_lat:.2f}s")

if __name__ == "__main__":
    main()