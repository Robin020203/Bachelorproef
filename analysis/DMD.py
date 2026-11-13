import numpy as np
from utils.audio_io import write_wav_sf, read_wav_sf
from utils.path import WAV_DIR
from utils.acoustics import eig_to_frequency, stereo_ratio_and_shift
from pydmd import HODMD

#help(HODMD)

# 1. Read WAV file
sample_rate, stereo_data = read_wav_sf(WAV_DIR / 'DRUM_AUDACITY.wav')
audio_matrix = stereo_data.T #(2 rows (channels), samples)

# 2. Applying Higher Order DMD (d dimensions)
hodmd = HODMD(svd_rank=0, svd_rank_extra=-1, d=50) # higher dimension (d=50)
hodmd.fit(audio_matrix)

# DEBUG
#print(f"Calculated DMD modi (d): {len(hodmd.modes)}")
#print(f"Modes shape (d, svd_rank): {hodmd.modes.shape}")
#print(f"Dynamics shape (svd_rank, sample_rate): {hodmd.dynamics.shape}")
#print(hodmd.modes)

eigs = hodmd.eigs
#print("Absolute eigenvalues:", np.abs(eigs))
frequencies = eig_to_frequency(eigs, sample_rate)
#print("Frequency (Hz):", frequencies)

# (2,8) = (2 x 88255) * (88255 x 8) because pinv = pseudo inverse
# multiplying by pseudo inverse = best solution
M = audio_matrix @ np.linalg.pinv(hodmd.dynamics)   # output: dim (2, 8)

# 3. Ratio and sample_shift for all eigenvalues big enough to be a pure tone
for i in range(len(eigs)):
    if np.abs(eigs[i]) > 0.999 and frequencies[i] > 0: # every real tone with positive frequency
        ratio, sample_shift = stereo_ratio_and_shift(M, eigs, i)
        f = frequencies[i]
        print("Frequency (Hz):", f)
        print("Sample shift:", sample_shift)
        print("Ratio: ", ratio)

# 4. Reconstruction of HODMD data
#print("HODMD output data: ", hodmd.reconstructed_data.real)
reconstructed_data = hodmd.reconstructed_data.real.T
write_wav_sf(WAV_DIR / 'Drum_hodmd2.wav', sample_rate, reconstructed_data)