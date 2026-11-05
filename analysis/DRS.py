import numpy as np
from utils.audio_io import write_wav_sf, read_wav_sf, split_stereo
from utils.path import WAV_DIR
from drs.core import drs, reconstruction, mirror, resonant_frequencies

# 1. Read WAV file and split stereo data
sample_rate, stereo_data = read_wav_sf(WAV_DIR / 'stereo_test6.wav')

left_channel, right_channel = split_stereo(stereo_data)

# 2. Use drs separately
window_length = 1024

# Output: list of (ds_zs, ds_zs_rev) per slice (samples test/window_length = 86)
drs_left = drs(left_channel, window_length)
drs_right = drs(right_channel, window_length)

# FIRST SLICE
# TODO: iterate over every slice, and put this in utils-function
slice_left, slice_right = drs_left[0], drs_right[0]

ds_zs_l, ds_zs_r = slice_left[0], slice_right[0]
ds_zl_rev_l, ds_zs_rev_r = drs_left[1], drs_right[1]

amplitudes_l, poles_l = np.unstack(ds_zs_l)
amplitudes_r, poles_r = np.unstack(ds_zs_r)

frequencies_l, frequencies_r = resonant_frequencies(poles_l, sample_rate), resonant_frequencies(poles_r, sample_rate)
#print(frequencies_l, frequencies_r)
#np.real(freq)
# TODO: find most dominant resonance(s) and calculate ratio and sample_shift



"""
# Reconstruction

reconstructed_left_list = []
for slice_item in drs_left:
    # Returns 4 elements (error)
    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
    reconstructed_left_list.append(rec_slice)
# Concatenate all reconstructed left slices
reconstructed_left = np.concatenate(reconstructed_left_list)

reconstructed_right_list = []
for slice_item in drs_right:
    # Returns 4 elements (error)
    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
    reconstructed_right_list.append(rec_slice)
# Concatenate all reconstructed right slices
reconstructed_right = np.concatenate(reconstructed_right_list)

# Same length (left & right)
min_len = min(len(reconstructed_left), len(reconstructed_right))
reconstructed_audio = np.column_stack((reconstructed_left[:min_len], reconstructed_right[:min_len]))

# Save
write_wav_sf(WAV_DIR / 'drs.wav', sample_rate, reconstructed_audio)
"""