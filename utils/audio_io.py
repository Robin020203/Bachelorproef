import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd


def read_wav16b(path):
    sample_rate, audio = wavfile.read(path)
    # Convert back from 16-bit PCM to float32 [-1,1] for calculations
    audio = audio.astype(np.float32) / 32767.0
    return sample_rate, audio

def write_wav16b(path, sample_rate, audio):
    # Convert float32 to 16-bit PCM (for WAV file)
    audio_int16 = (audio * 32767).astype(np.int16)
    # TODO: add clip function?
    wavfile.write(path, sample_rate, audio_int16)

def read_wav_sf(path):
    # sf.read returns float64, add dtype='float32' for float32 in function
    audio, sample_rate = sf.read(path)
    return sample_rate, audio

def write_wav_sf(path, sample_rate, audio):
    # Convert float to 16-bit PCM (for WAV file)
    clipped_audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, clipped_audio, sample_rate, subtype='PCM_16')

def split_stereo(audio):
    # splits multi channel audio into left and right channel
    if audio.ndim < 2:
        raise ValueError("wav file needs to be stereo")
    return audio[:, 0], audio[:, 1]

def resample(audio, from_rate, to_rate):
    # convert audio from 'from_rate' to 'to_rate'
    if from_rate == to_rate:
        return audio, to_rate
    else:
        g = gcd(from_rate, to_rate)
        audio2 = resample_poly(audio, to_rate // g, from_rate // g) # up, down
        return audio2, to_rate
