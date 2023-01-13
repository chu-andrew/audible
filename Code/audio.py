import numpy as np
import pyaudio
import time
from tqdm import tqdm


def play_frequency(Fs):
    VOLUME = 0.1  # range [0.0, 1.0]
    SAMPLE_RATE = 44100  # sampling rate, Hz, must be integer
    MOMENT_LENGTH = 0.25

    p = pyaudio.PyAudio()

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True)

    # play
    start_time = time.time()
    for f in tqdm(Fs):
        signal_i = signal(f, SAMPLE_RATE, MOMENT_LENGTH)
        bytes_i = (VOLUME * signal_i).tobytes()
        stream.write(bytes_i)
    end_time = time.time()
    print("Played sound for {:.2f} seconds".format(end_time - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()


def signal(moment_fs, sample_rate, moment_length):
    # generate samples, note conversion to float32 array

    samples = []
    for f in moment_fs:
        sample = (np.sin(2 * np.pi * np.arange(sample_rate * moment_length) * f / sample_rate)).astype(np.float32)
        samples.append(sample)

    combined = samples[0]
    for i in range(1, len(samples)):
        combined += samples[i]
    normalized = np.int16((combined / combined.max()) * 2 ** 15 - 1)  # normalize to 16-bit

    return normalized
