"""Text-to-speech helper using ElevenLabs SDK.

This module initializes its own ElevenLabs client from environment
variables to avoid circular imports. It decodes MP3 output using
`pydub` when available, falls back to `soundfile` if possible, and
saves the raw file for manual inspection if decoding is unavailable.
"""

import io
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from elevenlabs.client import ElevenLabs
except Exception:
    ElevenLabs = None
try:
    from elevenlabs.client import ElevenLabs
except Exception:
    ElevenLabs = None

try:
    import sounddevice as sd
except Exception:
    sd = None

import numpy as np

# Optional libraries for decoding
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    import soundfile as sf
except Exception:
    sf = None

# Initialize ElevenLabs client from env to avoid circular imports
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
eleven_client = None
if ElevenLabs is not None and ELEVENLABS_API_KEY:
    try:
        eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception:
        eleven_client = None


def speak(text: str):
    print("🔊 Speaking...")

    if eleven_client is None or not VOICE_ID:
        print("[TTS skipped] ElevenLabs client or VOICE_ID not configured")
        # Fallback: attempt local TTS engine (pyttsx3) so speech still works offline
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass
        return

    try:
        audio = eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            text=text,
            output_format="mp3_44100_128",
        )
    except Exception as e:
        print(f"[TTS error] ElevenLabs convert failed: {e}")
        # fallback to local TTS
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            try:
                import subprocess

                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak(\"{text.replace('"','\'') }\");",
                ]
                subprocess.run(cmd, check=False)
                return
            except Exception:
                return

    # Debug: show type of returned object for diagnostics
    try:
        print(f"[TTS debug] audio response type: {type(audio)}")
    except Exception:
        pass

    # Robustly extract bytes from many possible SDK response shapes
    audio_bytes = None
    try:
        # direct bytes
        if isinstance(audio, (bytes, bytearray)):
            audio_bytes = bytes(audio)
        # file-like object
        elif hasattr(audio, "read"):
            audio_bytes = audio.read()
        # requests.Response-like
        elif hasattr(audio, "content"):
            audio_bytes = getattr(audio, "content")
        # generator/iterable of chunks
        elif hasattr(audio, "iter_bytes"):
            audio_bytes = b"".join(audio.iter_bytes())
        else:
            # dict or mapping with common keys
            try:
                if isinstance(audio, dict):
                    for k in ("audio", "data", "content", "bytes"):
                        if k in audio and isinstance(audio[k], (bytes, bytearray)):
                            audio_bytes = bytes(audio[k])
                            break
                        if k in audio and isinstance(audio[k], str):
                            # base64? try to decode
                            try:
                                import base64

                                audio_bytes = base64.b64decode(audio[k])
                                break
                            except Exception:
                                pass
            except Exception:
                pass

            # attempt to join iterable; if it's a generator yielding chunks, iterate and coerce
            if audio_bytes is None:
                try:
                    # Special-case generator/iterable that yields chunks
                    import types

                    if isinstance(audio, types.GeneratorType) or hasattr(audio, "__iter__"):
                        parts = bytearray()

                        def _extract_bytes(obj):
                            # Try many heuristics to get bytes from an object
                            try:
                                if obj is None:
                                    return None
                                if isinstance(obj, (bytes, bytearray)):
                                    return bytes(obj)
                                if isinstance(obj, memoryview):
                                    return obj.tobytes()
                                if isinstance(obj, str):
                                    # base64 or plaintext
                                    import base64

                                    try:
                                        return base64.b64decode(obj)
                                    except Exception:
                                        return obj.encode("utf-8", errors="replace")
                                if hasattr(obj, "read"):
                                    try:
                                        return obj.read()
                                    except Exception:
                                        pass
                                if hasattr(obj, "content") and isinstance(getattr(obj, "content"), (bytes, bytearray)):
                                    return bytes(getattr(obj, "content"))
                                # common attribute names
                                for attr in ("audio", "data", "payload", "chunk", "bytes", "raw"):
                                    if hasattr(obj, attr):
                                        val = getattr(obj, attr)
                                        if isinstance(val, (bytes, bytearray)):
                                            return bytes(val)
                                        if isinstance(val, str):
                                            import base64

                                            try:
                                                return base64.b64decode(val)
                                            except Exception:
                                                return val.encode("utf-8", errors="replace")
                                # try __bytes__
                                try:
                                    b = bytes(obj)
                                    if b:
                                        return b
                                except Exception:
                                    pass
                            except Exception:
                                return None
                            return None

                        # collect and inspect first several chunks for debugging
                        first_chunks = []
                        for i, chunk in enumerate(audio):
                            if i < 8:
                                first_chunks.append((type(chunk), repr(chunk)[:200]))
                            try:
                                b = _extract_bytes(chunk)
                                if b:
                                    parts.extend(b)
                            except Exception:
                                continue
                        # if we captured parts, use them; otherwise save chunk reprs for debug
                        if parts:
                            audio_bytes = bytes(parts)
                        else:
                            try:
                                with open("eleven_chunks_debug.txt", "w", encoding="utf-8") as df:
                                    for t, r in first_chunks:
                                        df.write(f"TYPE: {t}\nREPR: {r}\n---\n")
                                print("[TTS debug] saved eleven_chunks_debug.txt containing first yielded chunk types/reprs")
                            except Exception:
                                pass
                    else:
                        audio_bytes = b"".join(audio)
                except Exception:
                    audio_bytes = None
    except Exception as e:
        print(f"[TTS debug] error while extracting bytes: {e}")

    if not audio_bytes:
        # save debug representation for inspection
        try:
            with open("eleven_response_debug.bin", "wb") as f:
                if isinstance(audio, str):
                    f.write(audio.encode("utf-8", errors="replace"))
                else:
                    try:
                        f.write(repr(audio).encode("utf-8", errors="replace"))
                    except Exception:
                        f.write(b"<unserializable-audio-response>")
            print("[TTS error] unable to obtain audio bytes from SDK response; saved eleven_response_debug.bin for inspection")
        except Exception as e:
            print(f"[TTS error] unable to obtain audio bytes and failed to save debug file: {e}")
        # Try local pyttsx3 fallback so assistant still speaks
        try:
            import pyttsx3

            print("[TTS fallback] using local pyttsx3 fallback")
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return
        except Exception as e:
            print(f"[TTS fallback pyttsx3 failed] {e}")
            # Try Windows PowerShell System.Speech fallback (works without extra Python deps)
            try:
                import subprocess, shlex

                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak(\"{text.replace('"','\'') }\");",
                ]
                subprocess.run(cmd, check=False)
                return
            except Exception as e2:
                print(f"[TTS fallback PowerShell failed] {e2}")
        return

    # Try decoding MP3 using pydub (preferred), then soundfile, else save to disk
    if AudioSegment is not None:
            try:
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

                # Optional downsampling / re-encoding to reduce file size for saved output
                try:
                    target_rate = int(os.getenv("TTS_TARGET_RATE", "22050"))
                except Exception:
                    target_rate = 22050
                try:
                    target_channels = int(os.getenv("TTS_TARGET_CHANNELS", "1"))
                except Exception:
                    target_channels = 1
                target_bitrate = os.getenv("TTS_TARGET_BITRATE", "64k")

                # Create a reduced-quality version for saving/export; use for playback to reduce CPU/IO
                reduced_seg = seg.set_frame_rate(target_rate).set_channels(target_channels)

                # Normalize based on sample width and convert to float32 numpy array for playback
                samples = np.array(reduced_seg.get_array_of_samples())
                if reduced_seg.channels > 1:
                    samples = samples.reshape((-1, reduced_seg.channels))
                max_val = float(1 << (8 * reduced_seg.sample_width - 1))
                audio_np = samples.astype(np.float32) / max_val

                if sd is None:
                    print("[Audio playback skipped] sounddevice not available")
                else:
                    sd.play(audio_np, samplerate=reduced_seg.frame_rate)
                    sd.wait()

                # Export reduced MP3 bytes for smaller on-disk files (if needed)
                try:
                    out_buf = io.BytesIO()
                    reduced_seg.export(out_buf, format="mp3", bitrate=target_bitrate)
                    out_buf.seek(0)
                    reduced_bytes = out_buf.read()
                    # overwrite audio_bytes with reduced mp3 for downstream saving
                    audio_bytes = reduced_bytes
                except Exception:
                    # If export fails, keep original audio_bytes
                    pass

                return
            except Exception as e:
                print(f"[TTS decode with pydub failed] {e}")

    if sf is not None:
        try:
            audio_data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if sd is None:
                print("[Audio playback skipped] sounddevice not available")
                return
            sd.play(audio_data, samplerate)
            sd.wait()
            return
        except Exception as e:
            print(f"[TTS decode with soundfile failed] {e}")

    # Fallback: save file for manual playback/inspection
    try:
        out_path = "eleven_out.mp3"
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        print(f"[Audio saved] {out_path} (install 'pydub' + ffmpeg or 'soundfile' to play automatically)")
    except Exception as e:
        print(f"[TTS save failed] {e}")
