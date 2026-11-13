import numpy as np
from pydmd import HODMD
from utils.audio_io import read_wav_sf, write_wav_sf
from utils.path import WAV_DIR

# 1. Read WAV file
input_file = 'DRUM_AUDACITY.wav'
sample_rate, stereo_data = read_wav_sf(WAV_DIR / input_file)
audio_matrix = stereo_data.T

# 2. hodmd
hodmd = HODMD(svd_rank=0, svd_rank_extra=-1, d=50) # higher dimension
hodmd.fit(audio_matrix)

eigs = hodmd.eigs
print("Absolute eigenvalues:", np.abs(eigs))
frequencies = np.angle(eigs) * sample_rate / (2 * np.pi) # f = ω/2π (* sample_rate)

M = audio_matrix @ np.linalg.pinv(hodmd.dynamics)
#L, R = M[0], M[1]
ratios = np.abs(M[1]) / np.abs(M[0])

print("Ratios :", ratios)

real_tones = []
for i in range(len(eigs)):
    if np.abs(eigs[i]) > 0.999 and frequencies[i] > 0: # every real tone with positive frequency
        real_tones.append(i)
        # M already defined above
        L, R = M[0, i], M[1, i]
        f = frequencies[i]
        fase_shift = np.angle(R/L)
        #print("DEBUG fase_shift:", fase_shift)
        #print("DEBUG angle eigenvalue:", np.angle(eigs[i]))
        #sample_shift = (fase_shift * sample_rate) / (2 * np.pi * f)
        sample_shift = fase_shift / np.angle(eigs[i])
        ratio = np.abs(R) / np.abs(L)
        print("Frequency (Hz):", f)
        #print("Sample shift:", sample_shift)
        print("Ratio: ", ratio)
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

    # All modes (+f and -f) with a ratio close to the center and strong enough
    members = np.where(
        (np.abs(ratios - center) / center < 0.05) &
        (np.abs(eigs) > 0.999)
    )[0]

    print(f"\nSource {g}: ratio ≈ {center:.3f}")
    print("  anchor tones (Hz):", np.round(frequencies[group], 1))
    print(f"  {len(members)} member modes, positive freqs (Hz):",
          np.round(np.sort(frequencies[members][frequencies[members] > 0]), 1))

    # Sum of only this source's modes: M[:, members] @ dynamics[members, :]
    source = (M[:, members] @ hodmd.dynamics[members, :]).real.T  # (samples, 2)

    out_file = WAV_DIR / f"{base_name}_source_{g}.wav"
    write_wav_sf(out_file, sample_rate, source)
    print("  saved:", out_file.name)
