import numpy as np
from utils.audio_io import write_wav16b, read_wav16b
from utils.path import WAV_DIR
from pydmd import HODMD

# 1. Read WAV file
sample_rate, stereo_data = read_wav16b(WAV_DIR / 'cello_stereo.wav')

# 2. Make an audio matrix (2 rows (channels), samples)
audio_matrix = stereo_data.T

# 3. Applying Higher Order DMD
hodmd = HODMD(svd_rank=0, d=10)
hodmd.fit(audio_matrix)

print(f"Calculated DMD modi (d): {len(hodmd.modes)}")
print(f"Modes shape (d, svd_rank): {hodmd.modes.shape}")
print(f"Dynamics shape (svd_rank, sample_rate): {hodmd.dynamics.shape}")
print(hodmd.modes)

# 4. Reconstruction of HODMD data
reconstructed_data = hodmd.reconstructed_data.real
reconstructed_stereo = reconstructed_data.T
write_wav16b(WAV_DIR / 'cello_hodmd_rec1.wav', sample_rate, reconstructed_stereo)