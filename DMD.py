import numpy as np
from scipy.io import wavfile
from pydmd import HODMD

# 1. Read WAV file
sample_rate, stereo_data = wavfile.read('stereo_test.wav')

# 2. Make an audio matrix (2 rows (channels), 100 timestamps)

audio_matrix = stereo_data.astype(np.float32) / 32767.0     # samples, channels(2)
audio_matrix = audio_matrix.T                               # channels(2), samples

# 3. Applying Higher Order DMD
hodmd = HODMD(svd_rank=0, d=10)
hodmd.fit(audio_matrix)

print(f"Calculated DMD modi (d): {len(hodmd.modes)}")
print(f"Modes shape (d, svd_rank): {hodmd.modes.shape}")
print(f"Dynamics shape (svd_rank, sample_rate): {hodmd.dynamics.shape}")
print(hodmd.modes)

# 4. Reconstruction of HODMD data
reconstructed_data = hodmd.reconstructed_data.real
reconstructed_stereo = (reconstructed_data.T * 32767).astype(np.int16)
wavfile.write('hodmd_reconstructed.wav', sample_rate, reconstructed_stereo)