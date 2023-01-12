import scipy
import numpy as np
from matplotlib import pyplot as plt
import audio
import encoding

if __name__ == '__main__':
    data_file = "../Data/empty.txt"

    SAMPLE_RATE = 44100  # Hertz
    THRESHOLD = 2500

    chunk_length = 4
    compression = False
    debug_print = False
    tones = encoding.get_tones(data_file, chunk_length, compression, debug_print)

    signal = np.array([])
    for t in tones:
        signal = np.concatenate((signal, audio.signal(t, SAMPLE_RATE, 0.25)))

    yf = scipy.fft.rfft(signal)
    xf = scipy.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)

    plt.plot(xf, np.abs(yf))
    plt.show()

    peaks, _ = scipy.signal.find_peaks(yf, THRESHOLD)
    print(peaks)
    print(len(peaks))
