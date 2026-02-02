import os
import io
from pathlib import Path
from dotenv import load_dotenv

import numpy as np

# Prefer using the local `speech.tts` helper which handles decoding/playback
from speech.tts import speak as tts_speak

try:
    from groq import Groq
except Exception:
    Groq = None

# 🔐 Load API keys from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CLIENT = None
if Groq is not None and GROQ_API_KEY:
    try:
        GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
    except Exception:
        GROQ_CLIENT = None


def get_llm_response(user_text: str) -> str:
    # Use a short system instruction for the LLM (do not inject the full SYSTEM_PROMPT.txt)
    # The SYSTEM_PROMPT.txt is reserved for identity and developer instructions only.
    messages = [
        {"role": "system", "content": "You are a concise, voice-first assistant that answers clearly and briefly."},
        {"role": "user", "content": user_text},
    ]

    if GROQ_CLIENT is None:
        return "[LLM unavailable - set GROQ_API_KEY and install Groq SDK]"

    completion = GROQ_CLIENT.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.7,
    )
    return completion.choices[0].message.content


def get_identity_statement() -> str:
    """Return a concise first-person identity statement derived from SYSTEM_PROMPT.txt.

    This reads the system prompt, finds a "You are ..." line, and converts it to
    a natural "I am ..." reply so the assistant doesn't echo instructions verbatim.
    """
    sys_prompt_path = Path(__file__).resolve().parent / "SYSTEM_PROMPT.txt"
    default = "I am Voice AI — a voice-first assistant."
    if not sys_prompt_path.exists():
        return default
    try:
        with open(sys_prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Prefer lines beginning with 'You are'
                if s.lower().startswith("you are"):
                    # Replace leading 'You are' with 'I am' and trim
                    return s.replace("You are", "I am", 1).strip()
                # Prefer lines referencing 'Voice AI'
                if "voice ai" in s.lower() or "voice-first" in s.lower():
                    return s
    except Exception:
        return default
    return default


def speak(text: str):
    print("🔊 Speaking...")

    audio_bytes = eleven_client.text_to_speech.convert(
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        text=text,
        output_format="mp3_44100_128"
    )

    audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
    sd.play(audio_data, samplerate)
    sd.wait()


def main():
    print("🎙️ Voice AI Assistant Ready (type 'exit' to stop)\n")

    while True:
        user_text = input("You: ")

        if user_text.lower() == "exit":
            print("Goodbye 👋")
            break

        # If user asks about identity, return the system prompt identity directly
        lowers = user_text.strip().lower()
        if "who are" in lowers or "who r" in lowers or "who are you" in lowers or "who u" in lowers:
            # Use the helper which converts the system prompt into a natural first-person reply
            response_text = get_identity_statement()
        else:
            print("🧠 Thinking...")
            response_text = get_llm_response(user_text)

        print("Bot:", response_text)
        # Use the robust TTS helper to play the response in background (full-duplex)
        try:
            import threading

            def _play(text):
                try:
                    tts_speak(text)
                except Exception as e:
                    print(f"[TTS playback failed] {e}")

            t = threading.Thread(target=_play, args=(response_text,), daemon=True)
            t.start()
        except Exception as e:
            print(f"[TTS thread spawn failed] {e}")


if __name__ == "__main__":
    main()
