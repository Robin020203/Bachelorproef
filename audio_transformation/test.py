import numpy as np
from utils.audio_io import write_wav16b
from utils.path import WAV_DIR

# sound settings
sample_rate = 44100     # Standard sample rate (44100 samples per sec)
duration = 2            # Seconds audio
frequency = 440         # 440 Hz (A4 note)

# Time array
t = np.linspace(0, duration, duration * sample_rate, endpoint=False)

# Sin wave
audio = np.sin(2 * np.pi * frequency * t)

write_wav16b(WAV_DIR / 'test.wav', sample_rate, audio)