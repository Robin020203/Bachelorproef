import numpy as np
from utils.audio_io import write_wav16b, read_wav16b, resample
from utils.path import WAV_DIR
from utils.acoustics import distance, decay, tdoa_samples, delay_and_decay_signal

# 1. Read WAV file
sample_rate, audio = read_wav16b(WAV_DIR / 'test.wav') # 440 Hz Sine wave (sample_rate = 44100)
sample_rate2, audio2 = read_wav16b(WAV_DIR / 'test2.wav') # 300 Hz Sine wave (sample_rate = 44100)

audio2, sample_rate2 = resample(audio2, sample_rate2, sample_rate) # same sample rate!

# 2. Positions (m)
s1_pos = [0.0, 0.0]
s2_pos = [0.5, 0.0]
source_pos1 = [2.0, 1.0]
source_pos2 = [-4.0, 2.0]

distance_a1 = distance(source_pos1, s1_pos) # distance s1 to source1
distance_b1 = distance(source_pos1, s2_pos) # distance s2 to source1

distance_a2 = distance(source_pos2, s1_pos) # distance s1 to source2
distance_b2 = distance(source_pos2, s2_pos) # distance s2 to source2

decay_a1 = decay(distance_a1)
decay_b1 = decay(distance_b1)

decay_a2 = decay(distance_a2)
decay_b2 = decay(distance_b2)

# 3. Calculate Time Difference of Arrival (in samples)
sample_delay1 = tdoa_samples(distance_a1, distance_b1, sample_rate)
sample_delay2 = tdoa_samples(distance_a2, distance_b2, sample_rate2)

left_delay1 = max(sample_delay1, 0)
right_delay1 = max(-sample_delay1, 0)

left_delay2 = max(sample_delay2, 0)
right_delay2 = max(-sample_delay2, 0)

# 4. Make signals left & right with delay
total_length1 = len(audio) + abs(sample_delay1)
total_length2 = len(audio2) + abs(sample_delay2)

left_channel1 = delay_and_decay_signal(audio, left_delay1, decay_a1, total_length1)
right_channel1 = delay_and_decay_signal(audio, right_delay1, decay_b1, total_length1)

left_channel2 = delay_and_decay_signal(audio2, left_delay2, decay_a2, total_length2)
right_channel2 = delay_and_decay_signal(audio2, right_delay2, decay_b2, total_length2)

# 5. Padding
max_length = max(total_length1, total_length2)
pad_len1 = max_length - total_length1
pad_len2 = max_length - total_length2

left_channel1 = np.pad(left_channel1, (0, pad_len1), mode='constant')
right_channel1 = np.pad(right_channel1, (0, pad_len1), mode='constant')
left_channel2 = np.pad(left_channel2, (0, pad_len2), mode='constant')
right_channel2 = np.pad(right_channel2, (0, pad_len2), mode='constant')

# 6. Sum channel input
left_channel = left_channel1 + left_channel2
right_channel = right_channel1 + right_channel2

# 7. 2D stereo-array and save
stereo_audio = np.column_stack((left_channel, right_channel))
write_wav16b(WAV_DIR / '2sine.wav', sample_rate, stereo_audio)