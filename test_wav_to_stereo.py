import numpy as np
from scipy.io import wavfile

# 1. Read WAV file
sample_rate, mono_audio = wavfile.read('test.wav')
# Convert back from 16-bit PCM to float [-1,1] for calculations
mono_audio = mono_audio.astype(np.float32) / 32767.0

# 2. Important parameters
c = 343.0                           # Speed of sound (m/s)
d = 0.50                            # Underlying distance between 2 sensors (m)
angle_deg = 45.0                    # Angle of source in degrees (0 = in front, 90 = right, -90 left)
angle_rad = np.radians(angle_deg)   # Angle converted to radians

# 3. Calculate Time Difference of Arrival
# Formula: delay = (distance between sensors * sin(angle)) / speed of sound
time_delay = (d * np.sin(angle_rad)) / c
sample_delay = int(np.round(time_delay * sample_rate)) # TDOA to discrete values

n_samples = len(mono_audio) # 44100 * 2
abs_delay = abs(sample_delay)
total_length = n_samples + abs_delay

# 4. Make signal left & right with delay
left_channel = np.zeros(total_length)
right_channel = np.zeros(total_length)

if sample_delay > 0:
    # Source right (left delayed)
    right_channel[:n_samples] = mono_audio
    left_channel[sample_delay:n_samples + sample_delay] = mono_audio
    #right_channel[:n_samples - sample_delay] = mono_audio[sample_delay:]
    #left_channel = mono_audio
elif sample_delay < 0:
    # Source left (right delayed)
    abs_delay = abs(sample_delay)
    left_channel[:n_samples] = mono_audio
    right_channel[abs_delay:n_samples + abs_delay] = mono_audio
    #left_channel[:n_samples - abs_delay] = mono_audio[abs_delay:]
    #right_channel = mono_audio
else:
    left_channel = mono_audio
    right_channel = mono_audio

# 5. 2D stereo-array
stereo_audio = np.column_stack((left_channel, right_channel))

# Convert to 16-bit PCM again (for WAV file)
stereo_int16 = (stereo_audio * 32767).astype(np.int16)

# 6. Save
wavfile.write('stereo_test.wav', sample_rate, stereo_int16)