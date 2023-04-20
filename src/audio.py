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
    stream = pa.open(format=protocol.pa_format,
                     channels=protocol.num_channels,
                     rate=protocol.sample_rate,
                     output=True)

    print("Play data tones.")
    start_time = time.time()
    for freq in tqdm(tones):
        signal_i = generate_signal(freq, protocol.sample_rate, protocol.moment_len)
        bytes_i = (protocol.volume * signal_i).tobytes()
        stream.write(bytes_i)
    end_time = time.time()
    print("Played data tones for {:.2f} seconds".format(end_time - start_time))

    stream.stop_stream()
    stream.close()


def visualize_signal(signal, sample_rate):
    print("Visualize signal and fourier transform.")
    plt.title("SIGNAL")
    plt.plot(signal)
    plt.show()

    yf = scipy.fft.rfft(signal)
    yf = np.abs(yf)
    xf = scipy.fft.rfftfreq(len(signal), 1 / sample_rate)

    peaks, _ = scipy.signal.find_peaks(yf)
    max_peak = peaks[np.argmax(yf[peaks])]  # Find the index from the maximum peak
    x_max = xf[max_peak]  # Find the x value from that index

    plt.title("FOURIER")
    plt.plot(xf, yf)
    plt.axvline(x=x_max, label=f"Peak: {x_max:.2f}", ls="dotted", color="r")
    plt.legend(loc='best')
    plt.show()


def generate_signal(moment_frequency, sample_rate, moment_length):
    """Generate part of the signal (moment), with the given frequency, rate, and length."""
    sample = (np.sin(2 * np.pi * np.arange(sample_rate * moment_length) * moment_frequency / sample_rate)).astype(
        np.float32)

    return sample


def receive(protocol, file_name, pa):
    stream = pa.open(protocol.sample_rate, protocol.num_channels, protocol.pa_format, input=True,
                     frames_per_buffer=protocol.chunk_len)

    print("Start recording.")
    start = time.time()

    audio_frames = []
    # total bits = Hz / chunk bits * s
    total_num_bits = int(protocol.sample_rate / protocol.chunk_len * protocol.recording_seconds)
    for i in range(0, total_num_bits):
        data = stream.read(protocol.chunk_len)
        audio_frames.append(data)

    end = time.time()
    print(f"Recorded for {(end - start):.2f}s.")

    stream.stop_stream()
    stream.close()

    '''
    # write data to wav file
    wave_file = wave.open(file_name, 'wb')
    wave_file.setsampwidth(pa.get_sample_size(protocol.pa_format))
    wave_file.setframerate(protocol.sample_rate)
    wave_file.setnchannels(protocol.num_channels)
    wave_file.writeframes(b''.join(audio_frames))
    wave_file.close()
    '''

    return audio_frames


def receive_wav(protocol, file_name, pa):
    wf = wave.open(file_name, 'rb')
    stream = pa.open(format=protocol.pa_format,
                     channels=protocol.num_channels,
                     rate=protocol.sample_rate,
                     output=True)

    chunk = 1024
    audio_frames = []
    wav_data = wf.readframes(chunk)
    while wav_data:
        wav_data = wf.readframes(chunk)
        audio_frames.append(wav_data)
    wf.close()
    stream.close()

    return audio_frames
