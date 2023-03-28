import pyaudio
import wave
import numpy as np
import time
from tqdm import tqdm

import encoding


def send(data_file, protocol, pa):
    tones = encoding.encode(data_file, protocol)
    play_frequency(tones, protocol, pa)


def play_frequency(tones, protocol, pa):
    """Play array of tones passed from encoding.encode()."""
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = pa.open(format=pyaudio.paFloat32,
                    channels=protocol.num_channels,
                    rate=protocol.sample_rate,
                    output=True)

    print("Play data.")
    start_time = time.time()
    for freq in tqdm(tones):
        signal_i = generate_signal(freq, protocol.sample_rate, protocol.moment_len)
        bytes_i = (protocol.volume * signal_i).tobytes()
        stream.write(bytes_i)
    end_time = time.time()
    print("Played sound for {:.2f} seconds".format(end_time - start_time))

    stream.stop_stream()
    stream.close()


def generate_signal(moment_frequency, sample_rate, moment_length):
    """Generate part of the signal (moment), with the given frequency, rate, and length, for all necessary channels."""
    # note conversion to float32 array
    samples = []
    for f in moment_frequency:
        sample = (np.sin(2 * np.pi * np.arange(sample_rate * moment_length) * f / sample_rate)).astype(np.float32)
        samples.append(sample)

    combined = samples[0]
    for i in range(1, len(samples)):
        combined += samples[i]
    normalized = np.int16((combined / combined.max()) * 2 ** 15 - 1)  # normalize to 16-bit

    return normalized


def receive(protocol, file_name, record_seconds, pa):
    FORMAT = pyaudio.paInt16

    stream = pa.open(protocol.sample_rate, protocol.num_channels, FORMAT, input=True,
                    frames_per_buffer=protocol.chunk_len)

    audio_frames = []
    print("Start recording.")
    start = time.time()

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
