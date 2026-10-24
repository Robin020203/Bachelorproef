import numpy as np
from utils.audio_io import read_wav_sf, write_wav_sf, split_stereo
from utils.path import WAV_DIR
from utils.acoustics import distance, decay, delay_samples, delay_and_decay_signal

# 1. Read WAV file
sample_rate, audio_data = read_wav_sf(WAV_DIR / 'cello.wav')

# Take left and right channels separately
input_left, input_right = split_stereo(audio_data)

# 2. Positions
s1_pos = [0.0, 0.0]         # sensor 1 position
s2_pos = [0.5, 0.0]         # sensor 2 position (0.5m difference x-as)
source_pos = [10.0, 1.0]     # source position

distance_a = distance(source_pos, s1_pos) # distance s1 to source
distance_b = distance(source_pos, s2_pos) # distance s2 to source

decay_a = decay(distance_a) # decay by 1/r
decay_b = decay(distance_b) # decay by 1/r

# 3. Delay channels separately (in samples)
left_delay = delay_samples(distance_a, sample_rate)
right_delay = delay_samples(distance_b, sample_rate)

# 4. Make signal left & right with delay
total_length = max(left_delay, right_delay) + len(audio_data)

left_channel = delay_and_decay_signal(input_left, left_delay, decay_a, total_length)
right_channel = delay_and_decay_signal(input_right, right_delay, decay_b, total_length)

# 5. 2D stereo-array
stereo_audio = np.column_stack((left_channel, right_channel))

# 6. Save
write_wav_sf(WAV_DIR / 'cello_stereo1.wav', sample_rate, stereo_audio)