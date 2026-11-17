import numpy as np
from utils.audio_io import write_wav_sf, read_wav_sf
from utils.path import WAV_DIR
from utils.acoustics import eig_to_frequency, stereo_ratio_and_shift
from pydmd import HODMD

"""
This file is temporary, trying to find the best fit for L and R
"""

sample_rate, stereo_data = read_wav_sf(WAV_DIR / 'DRUM_AUDACITY.wav')
audio_matrix = stereo_data.T #(2 rows (channels), samples)

dimension = 50
hodmd = HODMD(svd_rank=0, svd_rank_extra=-1, d=dimension)
hodmd.fit(audio_matrix)

eigs = hodmd.eigs
frequencies = eig_to_frequency(eigs, sample_rate)

print("------------------------------------------------------------------")
print("SOLUTION 1: L and R from calculated matrix M")
print("------------------------------------------------------------------")
# Original solution: 2x88255 @ 88255x8 = 2x8 matrix
M = audio_matrix @ np.linalg.pinv(hodmd.dynamics)
for i in range(len(eigs)):
    if np.abs(eigs[i]) > 0.999 and frequencies[i] > 0:
        L, R = M[0, i], M[1, i]
        fase_shift = np.angle(R / L)
        sample_shift = fase_shift / np.angle(eigs[i])
        ratio = np.abs(R) / np.abs(L)
        f = frequencies[i]
        print("Frequency (Hz):", f)
        print("Sample shift:", sample_shift)
        print("Ratio: ", ratio)
        #print("R :", R, "L :", L)
        print("--------------------------------------------")

print("------------------------------------------------------------------")
print("SOLUTION 2: L and R from dmd.modes using vectors")
print("------------------------------------------------------------------")

modes = hodmd.modes
half = modes.shape[0] // 2 #50
for i in range(len(eigs)):
    if np.abs(eigs[i]) > 0.999 and frequencies[i] > 0:
        L = []
        R = []
        for row in range(modes.shape[0]):
            if row % 2 == 0:
                L.append(modes[row, i])
            else:
                R.append(modes[row, i])
        #    if row < half:
        #        L.append(modes[row, i])
        #    else:
        #        R.append(modes[row, i])
        L, R = np.array(L), np.array(R) #complex vectors

        fase_shift = np.angle(np.vdot(L, R)) # Dot product between 'complex conjugate of L' and 'R'
        sample_shift = fase_shift / np.angle(eigs[i])
        ratio = np.linalg.norm(np.abs(R)) / np.linalg.norm(np.abs(L))
        f = frequencies[i]
        print("Frequency (Hz):", f)
        print("Sample shift:", sample_shift)
        print("Ratio: ", ratio)
        # print("R :", R, "L :", L)
        print("--------------------------------------------")

#L
#R
#L
#R