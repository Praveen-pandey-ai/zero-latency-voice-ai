import sounddevice as sd
import numpy as np

def play_audio_chunk(chunk):
    audio = np.frombuffer(chunk, dtype=np.int16)
    sd.play(audio, samplerate=24000)
    sd.wait()
