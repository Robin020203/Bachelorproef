import numpy as np
from scipy.io import wavfile

sample_rate = 44100  #Standard sample rate
t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)  # 2 seconds audio
audio = np.sin(2 * np.pi * 440 * t) # Generate a tone of 440 Hz (A4)

# Transfer to 16-bit PCM
audio_int16 = (audio * 32767).astype(np.int16)

# Save
wavfile.write('test.wav', sample_rate, audio_int16)