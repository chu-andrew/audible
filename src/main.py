from dataclasses import dataclass
import pyaudio

import audio


@dataclass
class Protocol:
    num_channels: int
    sample_rate: int
    chunk_len: int

    f_min: float
    f_max: float

    moment_len: float  # defines how long each tone will be broadcast for
    volume: float  # [0.0, 1.0], used for pyaudio output

    compressed: bool
    debug: bool


protocol = Protocol(1, 44100, 4, 1875.0, 2500, 0.25, 0.1, True, True)
data_file = "../data/hello_world.txt"
output_file = "../data/microphone.wav"
recording_seconds = 4

# made one global pyaudio because
# https://stackoverflow.com/questions/34993895/cant-record-more-than-one-wave-with-pyaudio-no-default-output-device
# unsure if this was actually the cause
pa = pyaudio.PyAudio()

'''
if input("['play'] > "):
    audio.transmit(data_file, protocol, pa)
'''
if input("[r]ecord > ") == "r":
    audio.receive(protocol, output_file, recording_seconds, pa)

pa.terminate()
