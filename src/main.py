import pyaudio

from transmitter import Transmitter
from receiver import Receiver
from protocol import Protocol

if __name__ == "__main__":
    protocol = Protocol()
    pa = pyaudio.PyAudio()

    data_file = "../data/lorem.txt"
    recording_file = "../data/example.wav"

    transmitter = Transmitter(protocol, pa)
    if input("[p]lay > ") == "p":
        transmitter.transmit(data_file)
    if input("[r]ecord > ") == "r":
        """
        frames = receiver.receive(protocol, recording_file, pa)
        decoding.decode(frames, protocol)
        """
    if input("[pl]ayback > ") == "pl":
        """
        frames = receiver.receive_wav(protocol, recording_file, pa)
        decoding.decode(frames, protocol)
        """
        r = Receiver(protocol, pa)
        r.simulate_receive_wav("../data/transmitter_hello_world.wav")

    pa.terminate()

# TODO
'''
simulated noise tests
band filter
add error correction
add protocol data to the front of transmission
add variable thresholding for fourier peaks
implement changing use of compression (turn on or off based on size change)
'''
