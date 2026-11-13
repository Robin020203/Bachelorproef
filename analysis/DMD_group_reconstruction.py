import numpy as np
from pydmd import HODMD
from utils.audio_io import read_wav_sf, write_wav_sf
from utils.path import WAV_DIR
from utils.acoustics import eig_to_frequency, stereo_ratio_and_shift

# 1. Read WAV file
input_file = 'DRUM_AUDACITY.wav'
sample_rate, stereo_data = read_wav_sf(WAV_DIR / input_file)
audio_matrix = stereo_data.T

# 2. hodmd
hodmd = HODMD(svd_rank=0, svd_rank_extra=-1, d=50) # higher dimension
hodmd.fit(audio_matrix)

eigs = hodmd.eigs
frequencies = eig_to_frequency(eigs, sample_rate)

M = audio_matrix @ np.linalg.pinv(hodmd.dynamics)
ratios, _ = stereo_ratio_and_shift(M, eigs, np.arange(len(eigs)))  # for all modes
#print("Ratios :", ratios)

real_tones = []
for i in range(len(eigs)):
    if np.abs(eigs[i]) > 0.999 and frequencies[i] > 0: # every real tone with positive frequency
        real_tones.append(i)
        ratio, sample_shift = stereo_ratio_and_shift(M, eigs, i)
        f = frequencies[i]
        #print("Frequency (Hz):", f)
        #print("Sample shift:", sample_shift)
        #print("Ratio: ", ratio)
real_tones = np.array(real_tones)
ordered_tones = real_tones[np.argsort(ratios[real_tones])]



#------------------------
# COPIED CODE!!!!! TEST
#------------------------

groups = [[ordered_tones[0]]]
for k in ordered_tones[1:]:
    first = groups[-1][0]
    if (ratios[k] - ratios[first]) / ratios[first] < 0.05: # 5%
        groups[-1].append(k)
    else:
        groups.append([k])


#------------------------
# COPIED CODE!!!!! TEST
#------------------------

base_name = input_file.rsplit('.', 1)[0]

for g, group in enumerate(groups):
    center = np.median(ratios[group])
    lo, hi = ratios[group].min(), ratios[group].max()   # group reach
    tol = 1e-6                                          # small error

    # All modes with ratio in reach of real_values (also negative f)
    members = []
    for i in range(len(eigs)):
        in_range = lo * (1 - tol) <= ratios[i] <= hi * (1 + tol)
        strong = np.abs(eigs[i]) > 0.999  # real tone
        if in_range and strong:
            members.append(i)
    members = np.array(members)

    print(f"\nSource {g}: ratio ≈ {center:.3f}")
    print("  anchor tones (Hz):", np.round(frequencies[group], 1))
    print(f"  {len(members)} member modes, positive freqs (Hz):",
          np.round(np.sort(frequencies[members][frequencies[members] > 0]), 1))

    # Sum of only this source's modes: M[:, members] @ dynamics[members, :]
    source = (M[:, members] @ hodmd.dynamics[members, :]).real.T  # (samples, 2)

    out_file = WAV_DIR / f"{base_name}_source_{g}.wav"
    write_wav_sf(out_file, sample_rate, source)
    print("  saved:", out_file.name)