import numpy as np
from utils.audio_io import write_wav16b, read_wav16b
from utils.path import WAV_DIR

# 1. Read WAV file
sample_rate, mono_audio = read_wav16b(WAV_DIR / 'test.wav')

# 2. Important parameters
c = 343.0                           # Speed of sound (m/s)

s1_pos = np.array([0.0, 0.0])       # sensor 1 position
s2_pos = np.array([0.5, 0.0])       # sensor 2 position (0.5m difference x-as)
source_pos = np.array([2.0, 1.0])   # source position

distance_a = np.linalg.norm(source_pos - s1_pos) # distance s1 to source
distance_b = np.linalg.norm(source_pos - s2_pos) # distance s2 to source

decay_a = 1.0 / distance_a #distance_a**2
decay_b = 1.0 / distance_b #distance_b**2

# 3. Calculate Time Difference of Arrival

# Old formula: time_delay = (d * np.sin(angle_rad)) / c
# New formula: time_delay = (distance between distances to sensors) / c
time_delay = (distance_a - distance_b) / c
sample_delay = int(time_delay * sample_rate) # TDOA to discrete values

n_samples = len(mono_audio) # 44100 * 2
total_length = len(mono_audio) + abs(sample_delay)

# 4. Make signal left & right with delay
left_channel = np.zeros(total_length)
right_channel = np.zeros(total_length)

left_delay = max(sample_delay, 0)
right_delay = max(-sample_delay, 0)
left_channel[left_delay:n_samples + left_delay] = mono_audio * decay_a
right_channel[right_delay:n_samples + right_delay] = mono_audio * decay_b


# 5. 2D stereo-array
stereo_audio = np.column_stack((left_channel, right_channel))

# 6. Save
write_wav16b(WAV_DIR / 'stereo_test6.wav', sample_rate, stereo_audio)