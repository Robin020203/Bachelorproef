import numpy as np
from utils.audio_io import write_wav_sf, read_wav_sf
from utils.path import WAV_DIR
from pydmd import HODMD

help(HODMD)

# 1. Read WAV file
#sample_rate, stereo_data = read_wav_sf(WAV_DIR / 'stereo_test6.wav')
#sample_rate, stereo_data = read_wav_sf(WAV_DIR / '2sine.wav')
sample_rate, stereo_data = read_wav_sf(WAV_DIR / 'DRUM_AUDACITY.wav')

#stereo_data = stereo_data[:44100] # 1 sec

# 2. Make an audio matrix (2 rows (channels), samples)
audio_matrix = stereo_data.T

# 3. Applying Higher Order DMD (10 dimensions)
#hodmd = HODMD(svd_rank=0, svd_rank_extra=-1, d=20) # original
hodmd = HODMD(svd_rank=0, svd_rank_extra=-1, d=50) # higher dimension
hodmd.fit(audio_matrix)

# DEBUG and definitions
print(f"Calculated DMD modi (d): {len(hodmd.modes)}")
print(f"Modes shape (d, svd_rank): {hodmd.modes.shape}")
print(f"Dynamics shape (svd_rank, sample_rate): {hodmd.dynamics.shape}")
#print(hodmd.modes)
#print("L:",hodmd.modes[0])
#print("R:",hodmd.modes[1])

eigs = hodmd.eigs
#print("Eigenvalues:",eigs)
print("Absolute eigenvalues:", np.abs(eigs))
frequencies = np.angle(eigs) * sample_rate / (2 * np.pi) # f = ω/2π (* sample_rate)
#print("Frequency (Hz):", frequencies)

# 4. Ratio and sample_shift for MAX eigenvalue (1)
i_max_eig = np.argmax(np.abs(eigs)) # Max eigenvalue index
#print("Max eig index:", i_max_eig)
f_max = np.abs(np.angle(eigs[i_max_eig])) * sample_rate / (2 * np.pi) # Frequency of max eigenvalue
#print("Max frequency (Hz):", f_max)
#print("L:",hodmd.modes[0,i_max_eig])
#print("R:",hodmd.modes[1,i_max_eig])
#print("L:",np.abs(hodmd.modes[0,i_max_eig]))
#print("R:",np.abs(hodmd.modes[1,i_max_eig]))

#L, R = hodmd.modes[0, i_max_eig], hodmd.modes[1, i_max_eig]

#dyn  = np.array([[1, 2, 3, 4]])          # (1 x 4)
#data = np.array([[2.1, 3.9, 6.0, 8.1]])  # (1 x 4)
#M = data @ np.linalg.pinv(dyn)
#print("test M: ", M)

# (2,8) = (2 x 88255) * (88255 x 8) because pinv = pseudo inverse
# multiplying by pseudo inverse = best solution
M = audio_matrix @ np.linalg.pinv(hodmd.dynamics)   # output: dim (2, 8)
L, R = M[0, i_max_eig], M[1, i_max_eig]

fase_shift = np.angle(R/L)
#sample_shift = (fase_shift * sample_rate) / (2 * np.pi * f_max)
sample_shift = fase_shift / np.angle(eigs[i_max_eig])
ratio = np.abs(R) / np.abs(L)
print("Ratio: ", ratio)
print("Sample shift:", sample_shift)


# 5. Ratio and sample_shift for all eigenvalues big enough to be a pure tone
# TODO: change argmax in every true signal (> 0.99) to calculate ratio and sample_shift
for i in range(len(eigs)):
    if np.abs(eigs[i]) > 0.999 and frequencies[i] > 0: # every real tone with positive frequency
        # M already defined above
        L, R = M[0, i], M[1, i]
        f = frequencies[i]
        fase_shift = np.angle(R/L)
        print("DEBUG fase_shift:", fase_shift)
        print("DEBUG angle eigenvalue:", np.angle(eigs[i]))
        #sample_shift = (fase_shift * sample_rate) / (2 * np.pi * f)
        sample_shift = fase_shift / np.angle(eigs[i])
        ratio = np.abs(R) / np.abs(L)
        print("Frequency (Hz):", f)
        print("Sample shift:", sample_shift)
        print("Ratio: ", ratio)

# 6. Reconstruction of HODMD data
print("HODMD output data: ", hodmd.reconstructed_data.real)
reconstructed_data = hodmd.reconstructed_data.real
reconstructed_stereo = reconstructed_data.T
write_wav_sf(WAV_DIR / 'Drum_hodmd.wav', sample_rate, reconstructed_stereo)