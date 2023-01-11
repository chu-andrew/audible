import time
import numpy as np
import pyaudio


def play_frequency(Fs, durations):
    # https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python/27978895#27978895

    p = pyaudio.PyAudio()

    volume = 0.5  # range [0.0, 1.0]
    sample_rate = 44100  # sampling rate, Hz, must be integer

    samples = generate_samples(Fs, durations, sample_rate)

    # per @yahweh comment explicitly convert to bytes sequence
    output_bytes = (volume * samples).tobytes()

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    # play. May repeat with different volume values (if done interactively)
    start_time = time.time()
    stream.write(output_bytes)
    end_time = time.time()
    print("Played sound for {:.2f} seconds".format(end_time - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()


def generate_samples(Fs, durations, sample_rate):
    samples = sample(Fs[0], durations[0], sample_rate)
    for f, duration in zip(Fs, durations):
        sample_i = sample(f, duration, sample_rate)
        samples = np.append(samples, sample_i)

    return samples


def sample(f, duration, rate):
    # generate samples, note conversion to float32 array
    return (np.sin(2 * np.pi * np.arange(rate * duration) * f / rate)).astype(np.float32)
