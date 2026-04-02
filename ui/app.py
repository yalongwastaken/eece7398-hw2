# @file    app.py
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Gradio-based voice Q&A UI. Loads ASR, LLM, and TTS models at
#          startup and exposes a web interface for recording questions,
#          viewing transcripts, and playing spoken responses. Supports
#          zero-shot voice blending via Kokoro embedding interpolation.
# @usage   python ui/app.py

# imports
import os
import sys
import time
import types
import warnings
import logging
import yaml
import numpy as np
import torch
import gradio as gr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.asr import ASR
from src.llm import LLM
from src.tts import TTS

# available Kokoro voices for the UI dropdowns
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael",
          "bf_emma", "bm_george"]

SAMPLE_RATE = 16000  # whisper expects 16kHz input


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


cfg = load_config()

# load all models at startup
print("[ui] loading models...")
asr = ASR(cfg["asr"])
llm = LLM(cfg["llm"])
tts = TTS(cfg["tts"])
print("[ui] models ready.")


def load_voice_tensor(voice_name: str) -> torch.Tensor:
    # download and return a Kokoro voice embedding tensor
    import huggingface_hub
    path = huggingface_hub.hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename=f"voices/{voice_name}.pt"
    )
    return torch.load(path, weights_only=True)


def get_voice(voice_a: str, voice_b: str, blend: float):
    # return pure voice_a or voice_b at extremes, otherwise interpolate embeddings
    if blend == 0.0:
        return voice_a
    if blend == 1.0:
        return voice_b
    va = load_voice_tensor(voice_a)
    vb = load_voice_tensor(voice_b)
    return (1 - blend) * va + blend * vb


def synthesize_with_voice(self, text: str, voice) -> tuple[np.ndarray, float]:
    # synthesize using a voice name string or a custom embedding tensor
    t0 = time.time()
    chunks = []
    for _, _, audio in self.pipeline(text, voice=voice, speed=1.0):
        chunks.append(audio)
    elapsed = time.time() - t0
    return np.concatenate(chunks), elapsed


# patch tts instance to support custom voice tensors
tts.synthesize_with_voice = types.MethodType(synthesize_with_voice, tts)


def process(audio, voice_a: str, voice_b: str, blend: float, history: list):
    if audio is None:
        return history, None, "", "", "No audio recorded."

    sr, data = audio

    # normalize to float32
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / sr)
        data = np.interp(np.linspace(0, len(data), n), np.arange(len(data)), data).astype(np.float32)

    # ASR
    text, asr_time = asr.transcribe(data)
    if not text:
        return history, None, "", "", "No speech detected."

    # LLM
    response, llm_time = llm.generate(text)
    if not response:
        return history, None, text, "", "No response generated."

    # TTS with selected or blended voice
    voice = get_voice(voice_a, voice_b, blend)
    speech, tts_time = tts.synthesize_with_voice(response, voice)

    total = asr_time + llm_time + tts_time
    status = f"ASR {asr_time:.2f}s  |  LLM {llm_time:.2f}s  |  TTS {tts_time:.2f}s  |  total {total:.2f}s"

    # append turn to conversation history
    history = history or []
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})

    return history, (cfg["tts"]["sample_rate"], speech), text, response, status


# build Gradio UI
with gr.Blocks(title="Voice Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Voice Q&A System")
    gr.Markdown("Record your question → get a transcription, text response, and spoken audio reply.")

    with gr.Row():
        # left column: voice settings and instructions
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### Voice Settings")
            voice_a = gr.Dropdown(VOICES, value="af_heart", label="Voice A")
            voice_b = gr.Dropdown(VOICES, value="am_adam", label="Voice B")
            blend = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Blend  (0 = A, 1 = B)")
            gr.Markdown("*Mix two voices to create a custom speaker.*")
            gr.Markdown("---")
            gr.Markdown("### How to use")
            gr.Markdown("1. Click **Record** and speak\n2. Click **Stop**\n3. Click **Ask**\n4. Listen to the reply")

        # right column: main interaction area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=380)

            gr.Markdown("### Press record, speak your question, then press stop and click Ask")
            with gr.Row():
                audio_in = gr.Audio(sources=["microphone"], type="numpy",
                                    label="Your Question", scale=2)
                with gr.Column(scale=1):
                    submit = gr.Button("Ask", variant="primary", size="lg")
                    discard = gr.Button("Discard recording", size="lg")
                    clear = gr.Button("Clear all context", variant="stop", size="lg")

            with gr.Row():
                transcript_box = gr.Textbox(label="You said", interactive=False, scale=1)
                response_box = gr.Textbox(label="Response text", interactive=False, scale=1)

            audio_out = gr.Audio(label="Spoken reply", autoplay=True)
            status = gr.Textbox(label="Timing", interactive=False, max_lines=1)

    # button callbacks
    submit.click(
        fn=process,
        inputs=[audio_in, voice_a, voice_b, blend, chatbot],
        outputs=[chatbot, audio_out, transcript_box, response_box, status]
    )
    discard.click(fn=lambda: None, outputs=[audio_in])
    clear.click(
        fn=lambda: ([], None, None, "", "", ""),
        outputs=[chatbot, audio_out, audio_in, transcript_box, response_box, status]
    )

if __name__ == "__main__":
    demo.launch(share=False)