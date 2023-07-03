import pyaudio
import wave
import time
from dataclasses import dataclass
from typing import Any

from protocol import Protocol
from src.decoder import Decoder


@dataclass
class Receiver:
    protocol: Protocol
    pa: Any

    decoder: Decoder = None

    def __post_init__(self):
        self.decoder = Decoder(self.protocol)

    def receive_mic(self):
        """Get audio frames from microphone."""

        def callback_mic(in_data, frame_count, time_info, status):
            self.decoder.write_to_buffer(in_data)
            return in_data, pyaudio.paContinue

        stream = self.pa.open(format=self.protocol.pa_format,
                              channels=self.protocol.num_channels,
                              rate=self.protocol.sample_rate,
                              frames_per_buffer=self.protocol.chunk_len,
                              input=True,
                              stream_callback=callback_mic)

        print("Start recording.")

        start = time.time()
        while stream.is_active() and (time.time() - start) < self.protocol.recording_seconds:
            time.sleep(0.001)

        end = time.time()
        print(f"Recorded for {(end - start):.2f}s.")

        stream.close()

        # write data to wav file
        file_name = "../data/example2.wav"
        wave_file = wave.open(file_name, 'wb')
        wave_file.setsampwidth(self.pa.get_sample_size(self.protocol.pa_format))
        wave_file.setframerate(self.protocol.sample_rate)
        wave_file.setnchannels(self.protocol.num_channels)
        wave_file.writeframes(b''.join(self.decoder.frames_buffer))
        wave_file.close()
        print(f"Recording successfully written to {file_name}.")

    def receive_wav(self, file_name):
        """Get audio frames from existing wav file."""

        def callback_wav(in_data, frame_count, time_info, status):
            frames = wf.readframes(frame_count)
            self.decoder.write_to_buffer(frames)
            return frames, pyaudio.paContinue

        wf = wave.open(file_name, 'rb')
        stream = self.pa.open(format=self.protocol.pa_format,
                              channels=self.protocol.num_channels,
                              rate=self.protocol.sample_rate,
                              output=True,
                              frames_per_buffer=4096,
                              stream_callback=callback_wav)

        while stream.is_active():
            time.sleep(0.001)

        self.decoder.write_to_buffer(flush_buffer=True)

        wf.close()
        stream.close()

    def simulate_receive_wav(self, file_name):
        wf = wave.open(file_name, 'rb')

        for i in range(wf.getnframes() // 47):
            frames = wf.readframes(47)
            self.decoder.write_to_buffer(frames)

        self.decoder.write_to_buffer(flush_buffer=True)

        wf.close()
