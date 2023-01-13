import scipy
import numpy as np
from matplotlib import pyplot as plt
import audio as ad
import encoding
import pyaudio
import wave
import sounddevice as sd
import soundfile as sf
import time


def receive2(sample_rate, num_channels, tones): #attempt to record and play simultaneously
    fs = sample_rate
    duration = 14 #seconds
    sd.playrec(ad.play_frequency(tones), sample_rate, channels = 2)
    #sd.play(myrecording2, fs)
    print("waitingggg")
    #sd.wait()
    #sd.play(myrecording, fs)
    print("audio complete")
  
    sd.stop()
def receive(record_seconds, rate, chunk,file_name):
    format = pyaudio.paInt16
    channels = 2

    audio = pyaudio.PyAudio()
    print("boom started")
    stream = audio.open(format=format, channels = channels, rate = rate, input = True, frames_per_buffer = chunk)

    audio_frames = []
    
    
    ad.play_frequency(tones)
    #while time.time() <= record_seconds:
    print("recording started")
    for i in range(0, int(rate/chunk*record_seconds)): 
        data = stream.read(chunk)
        audio_frames.append(data)

 
    print("recording ended")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    #commits data to wav file
    waveFile = wave.open(file_name, 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(format))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(audio_frames))
    waveFile.close()
    

    #samplesPerFrameOut = (sampleRateOut/SAMPLE_RATE)*samplesPerFrame


if __name__ == '__main__':
    data_file = "../Data/hello_world.txt"

    SAMPLE_RATE = 44100  # Hertz
    THRESHOLD = 2500

    chunk_length = 4
    channels = 2
    compression = False
    debug_print = False

    tones = encoding.get_tones(data_file, chunk_length, channels, compression, debug_print)

    signal = np.array([])
    for t in tones:
        signal = np.concatenate((signal, ad.signal(t, SAMPLE_RATE, 0.25)))

    yf = scipy.fft.rfft(signal)
    xf = scipy.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)

    plt.plot(xf, np.abs(yf))
    plt.show()

    peaks, _ = scipy.signal.find_peaks(yf, THRESHOLD)
    print(peaks)
    print(len(peaks))
    
    print("Type 'start' to start recording")
    if input() == 'start':  #initiates recording
        receive(14, 44100, 2, 'pit.wav')
        #receive2(44100, 2, tones)

    

