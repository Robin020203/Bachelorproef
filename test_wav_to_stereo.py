import numpy as np
from scipy.io import wavfile

# 1. Read file
sample_rate, mono_audio = wavfile.read('test.wav')

# TEST: Normalise to float for safe calculations
mono_audio = mono_audio.astype(np.float32) / 32767.0

# 2. Parameters
c = 343.0            # Speed of sound m/s
d = 0.50             # Underlying distance between 2 sensors (50 cm)
angle_deg = 45.0     # Angle of source in degrees (0 = in front, 90 = completely right)
angle_rad = np.radians(angle_deg)

# 3. Calculate Time Difference of Arrival
# Formula for straight line approximation:
# t_delay = (d * sin(theta)) / c
time_delay = (d * np.sin(angle_rad)) / c
sample_delay = int(np.round(time_delay * sample_rate))

# 4. Make signal left & right with delay
n_samples = len(mono_audio)
left_channel = np.zeros(n_samples)
right_channel = np.zeros(n_samples)

if sample_delay > 0:
    # Source right (left delayed)
    right_channel[:n_samples - sample_delay] = mono_audio[sample_delay:]
    left_channel = mono_audio
elif sample_delay < 0:
    # Source left (right delayed)
    abs_delay = abs(sample_delay)
    left_channel[:n_samples - abs_delay] = mono_audio[abs_delay:]
    right_channel = mono_audio
else:
    left_channel = mono_audio
    right_channel = mono_audio

# 5. 2D stereo-array
stereo_audio = np.column_stack((left_channel, right_channel))

# 16-bit PCM
stereo_int16 = (stereo_audio * 32767).astype(np.int16)

# 6. Save
wavfile.write('stereo_test.wav', sample_rate, stereo_int16)