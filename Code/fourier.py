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
    
    
    y = signal
    N = signal.length
    Y_k = np.fft.fft(y)[0:int(N/2)]/N 
    Y_k[1:] = 2*Y_k[1:] 
    Pxx = np.abs(Y_k) 
    f = SAMPLE_RATE*np.arange((N/2))/N 
    fig,ax = plt.subplots()
    plt.plot(f,Pxx)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylabel('Amplitude idk units lmao')
    plt.xlabel('Frequency hz')
    plt.show()
def send():
    samplesPerFrameOut = (sampleRateOut/SAMPLE_RATE)*samplesPerFrame
    
