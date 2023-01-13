from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import scipy
import time
import wave

import audio
import encoding


def receive(record_seconds, channels, chunk, rate, file_name):
    FORMAT = pyaudio.paInt16
    pa = pyaudio.PyAudio()
    stream = pa.open(rate, channels, FORMAT, input=True, frames_per_buffer=chunk)

    audio_frames = []
    print("recording started")
    start = time.time()
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        audio_frames.append(data)
    end = time.time()
    print(f"recording ended: {(end - start):.2f}s")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # commits data to wav file
    wave_file = wave.open(file_name, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(pa.get_sample_size(FORMAT))
    wave_file.setframerate(rate)
    wave_file.writeframes(b''.join(audio_frames))
    wave_file.close()


if __name__ == '__main__':
    data_file = "../Data/hello_world.txt"

    SAMPLE_RATE = 44100  # Hertz
    THRESHOLD = 2500

    chunk_length = 4
    channels = 1
    compression = False
    debug_print = False

    tones = encoding.get_tones(data_file, chunk_length, channels, compression, debug_print)

    signal = np.array([])
    for t in tones:
        signal = np.concatenate((signal, audio.signal(t, SAMPLE_RATE, 0.25)))

    '''
    plt.title("SIGNAL")
    plt.plot(signal)
    plt.show()
    '''

    yf = scipy.fft.rfft(signal)
    xf = scipy.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)

    '''
    plt.title("FOURIER SIGNAL")
    plt.plot(xf, np.abs(yf))
    plt.show()
    '''

    peaks, _ = scipy.signal.find_peaks(yf, THRESHOLD)
    print(peaks)
    print("# of Peaks:", len(peaks))

    if input('> ') == "start":  # initiates recording
        receive(14, channels, chunk_length, SAMPLE_RATE, "../Data/microphone.wav")
