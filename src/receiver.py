import wave
import time


# TODO transition to a continuous recording system with wrappers for optional time limits and wav writing
def receive(protocol, file_name, pa):
    """Get audio frames from microphone."""
    stream = pa.open(protocol.sample_rate,
                     protocol.num_channels,
                     protocol.pa_format,
                     input=True,
                     frames_per_buffer=protocol.chunk_len)

    print("Start recording.")
    start = time.time()

    audio_frames = []
    # total bits = Hz / chunk bits * s
    total_num_bits = int(protocol.sample_rate / protocol.chunk_len * (protocol.recording_seconds * 2))
    for i in range(0, total_num_bits):
        data = stream.read(protocol.chunk_len)
        audio_frames.append(data)

    end = time.time()
    print(f"Recorded for {(end - start):.2f}s.")

    stream.stop_stream()
    stream.close()

    # '''
    # write data to wav file
    wave_file = wave.open(file_name, 'wb')
    wave_file.setsampwidth(pa.get_sample_size(protocol.pa_format))
    wave_file.setframerate(protocol.sample_rate)
    wave_file.setnchannels(protocol.num_channels)
    wave_file.writeframes(b''.join(audio_frames))
    wave_file.close()
    print(f"Recording successfully written to {file_name}.")
    # '''

    return audio_frames


def receive_wav(protocol, file_name, pa):
    """Get audio frames from existing wav file."""
    wf = wave.open(file_name, 'rb')
    stream = pa.open(format=protocol.pa_format,
                     channels=protocol.num_channels,
                     rate=protocol.sample_rate,
                     output=True)

    chunk = 1024
    audio_frames = []
    wav_data = wf.readframes(chunk)
    while wav_data:
        wav_data = wf.readframes(chunk)
        audio_frames.append(wav_data)
    wf.close()
    stream.close()

    return audio_frames
