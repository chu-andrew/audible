import numpy as np
from dataclasses import dataclass
import wave
import time
from typing import Any
from tqdm import tqdm

from protocol import Protocol
from encoder import Encoder
from auxiliary_functions import generate_tone_maps


@dataclass
class Transmitter:
    protocol: Protocol
    pa: Any
    tone_map: dict = None

    def __post_init__(self):
        self.tone_map = generate_tone_maps(self.protocol)

    def transmit(self, data_file):
        encoder = Encoder(data_file, self.protocol, self.tone_map)
        tones = encoder.encode()
        self.play_tones(tones)

    def play_tones(self, tones):
        """Play array of tones passed from the Encoder."""

        # for paFloat32 sample values must be in range [-1.0, 1.0]
        stream = self.pa.open(format=self.protocol.pa_format,
                              channels=self.protocol.num_channels,
                              rate=self.protocol.sample_rate,
                              output=True)

        bytes = []

        print("Play data tones.")
        start_time = time.time()

        for freq in tqdm(tones):
            signal_i = self._generate_signal(freq)
            bytes_i = (self.protocol.volume * signal_i).tobytes()
            bytes.append(bytes_i)
            stream.write(bytes_i)

        end_time = time.time()
        print("Played data tones for {:.2f} seconds".format(end_time - start_time))

        self.write_tones(bytes)

        stream.stop_stream()
        stream.close()

    def write_tones(self, bytes):
        wave_file = wave.open(f"../data/transmitter_{time.strftime('%m%d-%H%M%S')}.wav", "wb")

        wave_file.setnchannels(self.protocol.num_channels)
        wave_file.setsampwidth(self.pa.get_sample_size(self.protocol.pa_format))
        wave_file.setframerate(self.protocol.sample_rate)
        wave_file.writeframes(b''.join(bytes))
        wave_file.close()

    def _generate_signal(self, moment_frequency):
        """Generate part of the signal with the given frequency, rate, and length."""
        length = self.protocol.sample_rate * self.protocol.moment_len
        sample = (np.sin(2 * np.pi * np.arange(length) * moment_frequency / self.protocol.sample_rate)
                  ).astype(np.float32)

        return sample
