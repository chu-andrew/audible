import time
import numpy as np
import pyaudio
from tqdm import tqdm


def play_frequency(Fs):
    VOLUME = 0.1  # range [0.0, 1.0]
    SAMPLE_RATE = 44100  # sampling rate, Hz, must be integer
    MOMENT_LENGTH = 0.25

    def _sample(f):
        sample = signal(f, SAMPLE_RATE, MOMENT_LENGTH)
        output_bytes = (VOLUME * sample).tobytes()

        return output_bytes

    p = pyaudio.PyAudio()

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True)

    # play
    start_time = time.time()
    for f in tqdm(Fs):
        sample_i = _sample(f)
        stream.write(sample_i)
    end_time = time.time()
    print("Played sound for {:.2f} seconds".format(end_time - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()


def signal(f, sample_rate, moment_length):
    # generate samples, note conversion to float32 array
    return (np.sin(2 * np.pi * np.arange(sample_rate * moment_length) * f / sample_rate)).astype(np.float32)
