from dataclasses import dataclass
import pyaudio

import audio


@dataclass
class Protocol:
    num_channels: int
    sample_rate: int
    chunk_len: int
    moment_len: float  # defines how long each tone will be broadcast for
    volume: float  # [0.0, 1.0], used for pyaudio output
    compressed: bool
    debug: bool


protocol = Protocol(1, 44100, 4, 0.25, 0.1, True, True)
data_file = "../Data/hello_world.txt"
output_file = "../Data/microphone.wav"
recording_length = 14

# made one global pyaudio because
# https://stackoverflow.com/questions/34993895/cant-record-more-than-one-wave-with-pyaudio-no-default-output-device
# unsure if this was actually the cause
pa = pyaudio.PyAudio()

if input("['play'] > ") == "play":
    audio.send(data_file, protocol, pa)
if input("['record'] > ") == "record":
    audio.receive(protocol, output_file, recording_length, pa)

pa.terminate()
'''
tones = encoding.encode(data_file, protocol)

signal = np.array([])
for t in tones:
    signal = np.concatenate((signal, audio.signal(t, protocol.sample_rate, 0.25)))

plt.title("SIGNAL")
plt.plot(signal)
plt.show()

yf = scipy.fft.rfft(signal)
xf = scipy.fft.rfftfreq(len(signal), 1 / protocol.sample_rate)

plt.title("FOURIER SIGNAL")
plt.plot(xf, np.abs(yf))
plt.show()

THRESHOLD = 2500
peaks, _ = scipy.signal.find_peaks(yf, THRESHOLD)
print(peaks)
print("# of Peaks:", len(peaks))
'''
