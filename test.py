import numpy as np
from scipy.io import wavfile

# sound settings
sample_rate = 44100     # Standard sample rate (44100 samples per sec)
duration = 2            # Seconds audio
frequency = 440         # 440 Hz (A4 note)

# Time array
t = np.linspace(0, duration, duration * sample_rate, endpoint=False)

# Sin wave
audio = np.sin(2 * np.pi * frequency * t)

# Convert to 16-bit PCM (for WAV file)
audio_int16 = (audio * 32767).astype(np.int16)

# Save
wavfile.write('test.wav', sample_rate, audio_int16)