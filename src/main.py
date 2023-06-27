import pyaudio

from transmitter import Transmitter
import receiver
import decoding
from protocol import Protocol

if __name__ == "__main__":
    protocol = Protocol()
    pa = pyaudio.PyAudio()

    data_file = "../data/empty.txt"
    recording_file = "../data/hello_world_with_nums.wav"

    transmitter = Transmitter(protocol, pa)
    if input("[p]lay > ") == "p":
        transmitter.transmit(data_file)
    if input("[r]ecord > ") == "r":
        frames = receiver.receive(protocol, recording_file, pa)
        decoding.decode(frames, protocol)
    if input("[pl]ayback > ") == "pl":
        frames = receiver.receive_wav(protocol, recording_file, pa)
        decoding.decode(frames, protocol)

    pa.terminate()

# TODO
'''
implement continuous listening from audio.receive()
add error correction
add protocol data to the front of transmission
add variable thresholding for fourier peaks
implement changing use of compression (turn on or off based on size change)
'''
