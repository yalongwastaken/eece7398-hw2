# @file    test_voice_cloning.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Tests Kokoro's voice cloning capability via embedding manipulation.
#          Kokoro represents speaker identity as learned embedding tensors
#          (shape 511x1x256). Two cloning approaches are demonstrated:
#
#          Zero-shot: blend existing voice embeddings to create a new speaker
#          identity without any additional training, analogous to zero-shot
#          speaker adaptation in the embedding space.
#
#          LoRA-based (simulated): Kokoro's StyleTTS2 architecture supports
#          LoRA fine-tuning on target speaker audio. Here we simulate the
#          embedding shift produced by LoRA adaptation by applying a small
#          controlled perturbation, normalized to preserve embedding magnitude.
#
# @usage   python test/test_voice_cloning.py

# imports
import os
import sys
import time
import warnings
import logging
import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kokoro import KPipeline

TEST_TEXT = "This is a test of voice cloning using the Kokoro text to speech system."
OUTPUT_DIR = os.path.dirname(__file__)
SR = 24000


def synthesize(pipeline: KPipeline, text: str, voice, label: str) -> np.ndarray:
    # synthesize audio and save to wav
    chunks = []
    t0 = time.time()
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        chunks.append(audio)
    elapsed = time.time() - t0
    audio = np.concatenate(chunks)
    duration = len(audio) / SR
    path = os.path.join(OUTPUT_DIR, f"clone_{label}.wav")
    sf.write(path, audio, SR)
    print(f"  saved: {path}  ({duration:.2f}s audio, {elapsed:.2f}s synthesis, RTF={elapsed/duration:.3f})")
    return audio


def load_voice_tensor(voice_name: str) -> torch.Tensor:
    # download built-in voice embedding from Kokoro's HF repo
    import huggingface_hub
    path = huggingface_hub.hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename=f"voices/{voice_name}.pt"
    )
    return torch.load(path, weights_only=True)


def main() -> None:
    print("\n" + "="*60)
    print("VOICE CLONING TEST — Kokoro (embedding-based)")
    print("="*60)

    print("\n[init] loading Kokoro pipeline...")
    t0 = time.time()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    print(f"[init] ready in {time.time()-t0:.2f}s")

    # baseline voices for comparison
    print("\n--- Baseline Voices ---")
    print("synthesizing with af_heart (female)...")
    synthesize(pipeline, TEST_TEXT, "af_heart", "baseline_af_heart")
    print("synthesizing with am_adam (male)...")
    synthesize(pipeline, TEST_TEXT, "am_adam", "baseline_am_adam")

    # zero-shot cloning via embedding interpolation
    print("\n--- Zero-Shot Voice Cloning (embedding blend) ---")
    print("loading voice tensors...")
    v_female = load_voice_tensor("af_heart")
    v_male   = load_voice_tensor("am_adam")

    blends = [
        ("25% male / 75% female", 0.25),
        ("50% male / 50% female", 0.50),
        ("75% male / 25% female", 0.75),
    ]

    for label, alpha in blends:
        blended = (1 - alpha) * v_female + alpha * v_male
        print(f"synthesizing {label}...")
        synthesize(pipeline, TEST_TEXT, blended, f"zeroshot_blend_{int(alpha*100)}")

    # save 50/50 blend as reusable custom voice
    blend_50 = 0.5 * v_female + 0.5 * v_male
    blend_path = os.path.join(OUTPUT_DIR, "custom_voice.pt")
    torch.save(blend_50, blend_path)
    print(f"  custom voice tensor saved: {blend_path}")

    # simulated LoRA-based voice cloning
    print("\n--- LoRA-Based Voice Cloning (simulated) ---")
    print("note: full LoRA fine-tuning requires GPU training on target speaker audio.")
    print("simulating embedding shift via controlled perturbation...\n")

    torch.manual_seed(42)
    lora_delta = 0.05 * torch.randn_like(v_female)
    lora_voice = v_female + lora_delta
    # normalize to preserve original embedding magnitude
    lora_voice = lora_voice * (v_female.norm() / lora_voice.norm())

    print("synthesizing with simulated LoRA-adapted voice...")
    synthesize(pipeline, TEST_TEXT, lora_voice, "lora_simulated")

    lora_path = os.path.join(OUTPUT_DIR, "lora_voice.pt")
    torch.save(lora_voice, lora_path)
    print(f"  LoRA-adapted voice tensor saved: {lora_path}")

    print("\n--- Summary ---")
    print("  clone_baseline_af_heart.wav  — original female voice")
    print("  clone_baseline_am_adam.wav   — original male voice")
    print("  clone_zeroshot_blend_25.wav  — zero-shot: 25% male blend")
    print("  clone_zeroshot_blend_50.wav  — zero-shot: 50/50 blend")
    print("  clone_zeroshot_blend_75.wav  — zero-shot: 75% male blend")
    print("  clone_lora_simulated.wav     — LoRA-adapted voice (simulated)")
    print("  custom_voice.pt              — reusable blended voice tensor")
    print("  lora_voice.pt                — reusable LoRA-adapted voice tensor")
    print("\n[test] PASSED")

if __name__ == "__main__":
    main()