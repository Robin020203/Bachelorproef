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