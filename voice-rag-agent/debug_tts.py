from importlib import reload
import speech.tts as tts
reload(tts)

print('Calling speak() for debug')
tts.speak('debug test run')
print('speak() returned')
