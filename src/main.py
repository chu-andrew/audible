import pyaudio

import audio
import decoding
from protocol import Protocol

if __name__ == '__main__':
    protocol = Protocol("../data/config.yaml")

    data_file = "../data/hello_world.txt"
    recording_file = "../data/hello_world_with_nums.wav"

    pa = pyaudio.PyAudio()

    if input("[p]lay > ") == "p":
        audio.transmit(data_file, protocol, pa)
    if input("[r]ecord > ") == "r":
        frames = audio.receive(protocol, recording_file, pa)
        decoding.decode(frames, protocol)
    if input("[pl]ayback > ") == "pl":
        frames = audio.receive_wav(protocol, recording_file, pa)
        decoding.decode(frames, protocol)

    pa.terminate()

# TODO
'''
add start and end markers to transmission
implement continuous listening from audio.receive()
add error correction
add protocol data to the front of transmission
add variable thresholding for fourier peaks
implement changing use of compression (turn on or off based on size change)
'''
