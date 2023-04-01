import pyaudio
import wave
import numpy as np
import time

import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm

import encoding


def transmit(data_file, protocol, pa):
    """Main function for encoding and transmitting data as sound."""
    tones = encoding.encode(data_file, protocol)
    play_frequency(tones, protocol, pa)


def play_frequency(tones, protocol, pa):
    """Play array of tones passed from encoding.encode()."""
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=protocol.num_channels,
                     rate=protocol.sample_rate,
                     output=True)

    print("Play data tones.")
    start_time = time.time()
    for freq in tqdm(tones):
        print(freq)
        signal_i = generate_signal(freq, protocol.sample_rate, protocol.moment_len)
        bytes_i = (protocol.volume * signal_i).tobytes()
        stream.write(bytes_i)
    end_time = time.time()
    print("Played data tones for {:.2f} seconds".format(end_time - start_time))

    stream.stop_stream()
    stream.close()


def visualize_signal(signal, sample_rate, threshold):
    print("Visualize signal.")
    plt.title("SIGNAL")
    plt.plot(signal)
    plt.show()

    yf = scipy.fft.rfft(signal)
    xf = scipy.fft.rfftfreq(len(signal), 1 / sample_rate)

    plt.title("FOURIER SIGNAL")
    plt.plot(xf, np.abs(yf))
    plt.show()

    peaks, _ = scipy.signal.find_peaks(yf, threshold)
    print("Peaks:", peaks)
    print("# of Peaks:", len(peaks))


def generate_signal(moment_frequencies, sample_rate, moment_length):
    """Generate part of the signal (moment), with the given frequency, rate, and length, for all necessary channels."""
    # TODO add multi channel support: add and normalize
    # https://stackoverflow.com/questions/64958186/numpy-generate-sine-wave-signal-with-time-varying-frequency
    # https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
    # https://realpython.com/python-scipy-fft/
    sample = (np.sin(2 * np.pi * np.arange(sample_rate * moment_length) * moment_frequencies[0] / sample_rate)).astype(
        np.float32)

    return sample


def receive(protocol, file_name, record_seconds, pa):
    FORMAT = pyaudio.paInt16
    stream = pa.open(protocol.sample_rate, protocol.num_channels, FORMAT, input=True,
                     frames_per_buffer=protocol.chunk_len)

    print("Start recording.")
    start = time.time()

    audio_frames = []
    # total bits = Hz / chunk bits * s
    for i in range(0, int(protocol.sample_rate / protocol.chunk_len * record_seconds)):
        data = stream.read(protocol.chunk_len)
        audio_frames.append(data)

    end = time.time()
    print(f"Recorded for {(end - start):.2f}s.")

    stream.stop_stream()
    stream.close()

    # write data to wav file
    wave_file = wave.open(file_name, 'wb')
    wave_file.setsampwidth(pa.get_sample_size(FORMAT))
    wave_file.setframerate(protocol.sample_rate)
    wave_file.setnchannels(protocol.num_channels)
    wave_file.writeframes(b''.join(audio_frames))
    wave_file.close()

    # convert stream to np array
    stream_int = convert_bytes_stream(audio_frames)
    # print(len(audio_frames))
    # visualize_signal(stream_int, protocol, 2500)

    # TODO read https://www.samproell.io/posts/signal/peak-finding-python-js/
    # perform fft on sliding window of np array to retrieve frequencies # TODO change window and loop lengths
    window_frequencies = np.array([])
    for i in range(0, int(protocol.sample_rate / protocol.chunk_len * record_seconds),
                   int(protocol.sample_rate / protocol.chunk_len * protocol.moment_len / 2)):
        window = stream_int[i:i + int(protocol.sample_rate / protocol.chunk_len * protocol.moment_len / 2)]
        frequency = find_window_peak(window, protocol.sample_rate)
        window_frequencies = np.append(window_frequencies, frequency)


def convert_bytes_stream(frames_bytes):
    """Convert the bytes data from the PyAudio stream into a numpy array."""
    frames_int = np.array([])
    for bytes in frames_bytes:
        frames_int_i = np.frombuffer(bytes, dtype=np.int16)
        frames_int = np.concatenate((frames_int, frames_int_i))
    return frames_int


def find_window_peak(window, sample_rate):
    """Find most important frequency in given window of stream."""
    xf = scipy.fft.rfftfreq(len(window), 1 / sample_rate)
    yf = scipy.fft.rfft(window)
    peaks, properties = scipy.signal.find_peaks(yf)

    max_height_peak = np.nan
    if len(peaks > 0):
        max_height = properties["peak_heights"].max()
        for peak, height in zip(peaks, properties["peak_heights"]):
            if height == max_height: max_height_peak = peak  # TODO may cause problems if multiple peaks with same height

        plt.title(f"FOURIER SIGNAL {max_height_peak}")
        plt.plot(xf, np.abs(yf))
        plt.show()

        # print(peaks, max_height_peak)

    return max_height_peak
