# https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python/27978895#27978895


import time
import numpy as np
import pyaudio


def play_frequency(f_and_durations):
    p = pyaudio.PyAudio()

    volume = 0.5  # range [0.0, 1.0]
    fs = 44100  # sampling rate, Hz, must be integer
    # duration = 0.5  # in seconds, may be float
    # f = 440.0  # sine frequency, Hz, may be float

    samples = generate_samples(f_and_durations, fs)

    # per @yahweh comment explicitly convert to bytes sequence
    output_bytes = (volume * samples).tobytes()

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    # play. May repeat with different volume values (if done interactively)
    start_time = time.time()
    stream.write(output_bytes)
    end_time = time.time()
    print("Played sound for {:.2f} seconds".format(end_time - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()


def generate_samples(f_and_durations, fs):

    samples = (np.sin(2 * np.pi * np.arange(fs * f_and_durations[0][1]) * f_and_durations[0][0] / fs)).astype(np.float32)
    for f, duration in f_and_durations:
        # generate samples, note conversion to float32 array
        sample_i = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
        samples = np.append(samples, sample_i)

    return samples


if __name__ == '__main__':

    play_frequency([
        [350, 1],
        [360, 0.5],
        [370, 0.5],
        [380, 0.5],
        [390, 0.5],
        [400, 1],
    ])
