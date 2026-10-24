import numpy as np
from utils.audio_io import write_wav16b, read_wav16b, split_stereo
from utils.path import WAV_DIR
from drs.core import drs, reconstruction #, mirror, resonant_frequency

# 1. Read WAV file
sample_rate, stereo_data = read_wav16b(WAV_DIR / 'cello_stereo.wav')

left_channel, right_channel = split_stereo(stereo_data)

# 2. DRS parameters
window_length = 1024

#drs_slices = drs(audio_matrix.T, window_length)
drs_slices_left = drs(left_channel, window_length)
drs_slices_right = drs(right_channel, window_length)

# 3. Reconstruct all slices for het left channel
reconstructed_left_list = []
for slice_item in drs_slices_left:
    # Returns 4 elements (error)
    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
    reconstructed_left_list.append(rec_slice)

# Concatenate all reconstructed left slices
reconstructed_left = np.concatenate(reconstructed_left_list)

# 3. Reconstruct all slices for het right channel
reconstructed_right_list = []
for slice_item in drs_slices_right:
    # Returns 4 elements (error)
    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
    reconstructed_right_list.append(rec_slice)

# Concatenate all reconstructed right slices
reconstructed_right = np.concatenate(reconstructed_right_list)

# 3. Reconstruct all slices for channels together
#reconstructed_list = []
#for slice_item in drs_slices:
#    # Returns 4 elements (error)
#    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
#    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
#    reconstructed_list.append(rec_slice)

# Concatenate all reconstructed right slices
# reconstructed_total = np.concatenate(reconstructed_list)

# Same length (Left & Right)
min_len = min(len(reconstructed_left), len(reconstructed_right))
reconstructed_audio = np.column_stack((reconstructed_left[:min_len], reconstructed_right[:min_len]))

# 5. 16-bit PCM and save
write_wav16b(WAV_DIR / 'cello_drs_rec1.wav', sample_rate, reconstructed_audio)

#write_wav16b(WAV_DIR / 'drs_reconstructed_together.wav', sample_rate, reconstructed_audio)