import numpy as np
from scipy.io import wavfile
from drs.core import drs, reconstruction #, mirror, resonant_frequency

# 1. Read WAV file
sample_rate, stereo_data = wavfile.read('wav/cello_stereo.wav')

# 2. Make an audio matrix (channels, timestamps)

audio_matrix = stereo_data.astype(np.float32) / 32767.0
audio_matrix = audio_matrix.T

left_channel = audio_matrix[0]
right_channel = audio_matrix[1]

# 3. DRS parameters
window_length = 1024

#drs_slices = drs(audio_matrix, window_length)
drs_slices_left = drs(left_channel, window_length)
drs_slices_right = drs(right_channel, window_length)

# 4. Reconstruct all slices for het left channel
reconstructed_left_list = []
for slice_item in drs_slices_left:
    # Returns 4 elements (error)
    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
    reconstructed_left_list.append(rec_slice)

# Concatenate all reconstructed left slices
reconstructed_left = np.concatenate(reconstructed_left_list)

# 5. Reconstruct all slices for het left channel
reconstructed_right_list = []
for slice_item in drs_slices_right:
    # Returns 4 elements (error)
    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
    reconstructed_right_list.append(rec_slice)

# Concatenate all reconstructed right slices
reconstructed_right = np.concatenate(reconstructed_right_list)

# Reconstruct all slices for channels together
#reconstructed_list = []
#for slice_item in drs_slices:
#    # Returns 4 elements (error)
#    ds_zs_fwd, ds_zs_rev, _, _ = slice_item
#    rec_slice = reconstruction(ds_zs_fwd, ds_zs_rev, window_length)
#    reconstructed_right_list.append(rec_slice)

# Concatenate all reconstructed right slices
# reconstructed_total = np.concatenate(reconstructed_list)

# Same length
min_len = min(len(reconstructed_left), len(reconstructed_right))
reconstructed_audio = np.column_stack((reconstructed_left[:min_len], reconstructed_right[:min_len]))

# 6. 16-bit PCM and save
stereo_int16 = (reconstructed_audio * 32767).astype(np.int16)
wavfile.write('wav/cello_drs_rec.wav', sample_rate, stereo_int16)

#stereo_int16 = (reconstructed_total * 32767).astype(np.int16)
#wavfile.write('wav/drs_reconstructed_together.wav', sample_rate, stereo_int16)