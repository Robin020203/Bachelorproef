import numpy as np
from utils.audio_io import read_wav_sf, write_wav_sf, split_stereo
from utils.path import WAV_DIR

# 1. Read WAV file
sample_rate, audio_data = read_wav_sf(WAV_DIR / 'cello.wav')

# Take left and right channels separately
input_left, input_right = split_stereo(audio_data)

# 2. Important parameters
c = 343.0                           # Speed of sound (m/s)
s1_pos = np.array([0.0, 0.0])       # sensor 1 position (left sensor)
s2_pos = np.array([2.0, 0.0])       # sensor 2 position (right sensor)
source_pos = np.array([10.0, 1.0])   # source position

# distances and decay
distance_a = np.linalg.norm(source_pos - s1_pos)
distance_b = np.linalg.norm(source_pos - s2_pos)

decay_a = 1.0 / distance_a
decay_b = 1.0 / distance_b

# 3. TDOA separately
time_a = distance_a / c
time_b = distance_b / c

delay_samples_a = int(time_a * sample_rate)
delay_samples_b = int(time_b * sample_rate)

n_samples = len(audio_data)
max_delay = max(delay_samples_a, delay_samples_b)
total_length = max_delay + n_samples

# 4. Make signal left & right with delay
left_channel = np.zeros(total_length)
right_channel = np.zeros(total_length)

left_channel[delay_samples_a : delay_samples_a + n_samples] += input_left * decay_a
right_channel[delay_samples_b : delay_samples_b + n_samples] += input_right * decay_b

# 5. 2D stereo-array
stereo_audio = np.column_stack((left_channel, right_channel))

# 6. Save
write_wav_sf(WAV_DIR / 'cello_stereo1.wav', sample_rate, stereo_audio)