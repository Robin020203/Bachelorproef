import numpy as np

SpeedOfSound = 343.0 # m/s

def distance(pos_1, pos_2):
    return np.linalg.norm(np.asarray(pos_1) - np.asarray(pos_2))

def decay(dist, exponent=1):
    # decay by 1/r (standard) or 1/r^2 (exponential)
    return 1.0 / dist**exponent

def delay_samples(dist, sample_rate, c=SpeedOfSound):
    # Delay for 1 distance (in samples)
    return int((dist / c) * sample_rate)

def tdoa_samples(dist_1, dist_2, sample_rate, c=SpeedOfSound):
    # Time Difference of Arrival: Delay for difference between 2 distances (in samples)
    return int(((dist_1 - dist_2) / c) * sample_rate)

def delay_and_decay_signal(signal, delay, decay, total_length):
    n_samples = len(signal)
    channel = np.zeros(total_length)
    channel[delay:n_samples + delay] = signal * decay
    return channel

def eig_to_frequency(eigs, sample_rate):
    # Calculate frequencies from eigs: f = ω/2π (* sample_rate)
    return np.angle(eigs) * sample_rate / (2 * np.pi)

def stereo_ratio_and_shift(M, eigs, i):
    # Return ratio and sample shift for eigenvalue i
    L, R = M[0, i], M[1, i]
    fase_shift = np.angle(R / L)
    # sample_shift = (fase_shift * sample_rate) / (2 * np.pi * f)
    sample_shift = fase_shift / np.angle(eigs[i])
    ratio = np.abs(R) / np.abs(L)
    return ratio, sample_shift

def group_by_ratio(real_tones_array, ratios, relative_tol=0.05):
    # Group ratios based on tolerance
    ordered_tones = real_tones_array[np.argsort(ratios[real_tones_array])]
    groups = []
    for k in ordered_tones:

        if groups:
            first = groups[-1][0]
            # Boolean t/f
            same_group = (ratios[k] - ratios[first]) / ratios[first] < relative_tol
        else:
            same_group = False # DEBUG (first tone has no group)

        if same_group:
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups

def group_members(group, ratios, eigs, tol=1e-6):
    lo, hi = ratios[group].min(), ratios[group].max()  # group reach
    # All modes with ratio in reach of real_values (also negative f)
    members = []
    for i in range(len(eigs)):
        # boolean t/f
        in_range = lo * (1 - tol) <= ratios[i] <= hi * (1 + tol)
        if in_range and np.abs(eigs[i]) > 0.999:
            members.append(i)
    return np.array(members)