from speech import tts

def inspect():
    if tts.eleven_client is None or not tts.VOICE_ID:
        print('eleven client not configured')
        return
    gen = tts.eleven_client.text_to_speech.convert(
        voice_id=tts.VOICE_ID,
        model_id='eleven_multilingual_v2',
        text='hello world inspect',
        output_format='mp3_44100_128',
    )
    print('got', type(gen))
    for i, chunk in enumerate(gen):
        print(i, type(chunk), repr(chunk)[:200])
        if i >= 10:
            break

if __name__ == '__main__':
    inspect()
