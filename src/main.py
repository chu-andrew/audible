from dataclasses import dataclass

import numpy as np
import pyaudio

import audio
import decoding


@dataclass
class Protocol:
    num_channels: int
    sample_rate: int
    chunk_len: int
    recording_seconds: float

    f_min: float
    f_max: float

    np_format: np.dtype
    pa_format: int

    moment_len: float  # defines how long each tone will be broadcast for
    volume: float  # [0.0, 1.0], used for pyaudio output

    compressed: bool
    debug: bool


if __name__ == '__main__':

    protocol = Protocol(1, 44100, 4, 8,
                        1875.0, 2500,
                        np.dtype(np.single), pyaudio.paFloat32,
                        0.25, 0.1,
                        True, True)

    data_file = "../data/hello_world.txt"
    recording_file = "../data/microphone.wav"

    pa = pyaudio.PyAudio()

    if input("[p]lay > ") == "p":
        audio.transmit(data_file, protocol, pa)
    if input("[r]ecord > ") == "r":
        audio.receive(protocol, recording_file, pa)
    if input("[pl]ayback > ") == "pl":
        frames = audio.receive_wav(protocol, recording_file, pa)
        decoding.decode(frames, protocol)

    pa.terminate()
