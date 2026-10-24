import numpy as np
from utils.audio_io import write_wav16b, read_wav16b
from utils.path import WAV_DIR
from utils.acoustics import distance, decay, tdoa_samples, delay_and_decay_signal

# 1. Read WAV file
sample_rate, audio = read_wav16b(WAV_DIR / 'test.wav')

# 2. Positions (m)
s1_pos = [0.0, 0.0]
s2_pos = [0.5, 0.0]
source_pos = [2.0, 1.0]

distance_a = distance(source_pos, s1_pos) # distance s1 to source
distance_b = distance(source_pos, s2_pos) # distance s2 to source

decay_a = decay(distance_a)
decay_b = decay(distance_b)

# 3. Calculate Time Difference of Arrival (in samples)
sample_delay = tdoa_samples(distance_a, distance_b, sample_rate)

left_delay = max(sample_delay, 0)
right_delay = max(-sample_delay, 0)

# 4. Make signal left & right with delay
total_length = len(audio) + abs(sample_delay)

left_channel = delay_and_decay_signal(audio, left_delay, decay_a, total_length)
right_channel = delay_and_decay_signal(audio, right_delay, decay_b, total_length)

# 5. 2D stereo-array
stereo_audio = np.column_stack((left_channel, right_channel))

# 6. Save
write_wav16b(WAV_DIR / 'stereo_test6.wav', sample_rate, stereo_audio)