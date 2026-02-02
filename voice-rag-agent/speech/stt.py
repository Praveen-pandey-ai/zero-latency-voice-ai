import os
import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")


def start_transcriber(callback):
    transcriber = aai.RealtimeTranscriber(
        sample_rate=16000,
        on_data=lambda t: callback(t.text)
    )
    transcriber.connect()
    return transcriber
