import numpy as np
import scipy
import time
from matplotlib import pyplot as plt

import encoding
import auxiliary_functions


def decode(frames, protocol):
    signal = convert_bytes_stream(frames)
    f, t, Zxx = short_time_fourier(signal, protocol)
    freqs = fourier_peaks(f, t, Zxx)

    sampled_bitstring = process_tones(freqs, protocol)
    output_txt_file(sampled_bitstring, protocol.compressed)


def convert_bytes_stream(frames_bytes):
    """Convert the bytes data from the PyAudio stream into a numpy array."""
    frames_int = np.array([])
    for bytes in frames_bytes:
        frames_int_i = np.frombuffer(bytes, dtype=np.single)
        frames_int = np.concatenate((frames_int, frames_int_i))

    return frames_int


def short_time_fourier(signal, protocol):
    """Run STFT on entirety of captured signal."""
    bits_per_second = protocol.sample_rate / protocol.chunk_len
    segment_size = int(bits_per_second * protocol.moment_len)

    f, t, Zxx = scipy.signal.stft(signal, protocol.sample_rate, nperseg=segment_size)
    Zxx = np.abs(Zxx)

    # '''
    plt.pcolormesh(t, f, Zxx, shading="gouraud")
    plt.title("STFT Magnitude")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()
    # '''

    return f, t, Zxx


def fourier_peaks(f, t, Zxx):
    """From the STFT data, find and filter relevant area using peak heights."""
    Zxx = np.transpose(Zxx)  # flip rows/cols to get each sample's fft in each row

    frequencies = []
    heights = []
    for sample_stft in Zxx:
        peaks, properties = scipy.signal.find_peaks(
            sample_stft, height=(None, None))  # heights parameter necessary to get properties
        max_peak_index = np.argmax(sample_stft[peaks])  # Find the index of the maximum peak within peak array

        freq = f[peaks[max_peak_index]]
        max_height = properties["peak_heights"][max_peak_index]

        frequencies.append(freq)
        heights.append(max_height)

    peaks = filter_peaks(frequencies, heights, t)
    return peaks


def filter_peaks(frequencies, heights, t):
    """Remove all peaks of insufficient height (strength of signal) and return area of interest."""
    # FIXME: JJ
    # TODO this is a naive method that may lead to 0s within data area
    MIN_PEAK_HEIGHT = 0.001  # determined with stft heights plot # TODO is there a way to not use a constant?
    filtered = frequencies.copy()
    for i in range(len(filtered)):
        if heights[i] < MIN_PEAK_HEIGHT:
            filtered[i] = 0

    # '''
    # View all sampled tones, the tones with fft peaks above threshold, and fft peak heights.
    fig, axs = plt.subplots(2)
    axs[0].plot(t, frequencies, "go")
    axs[0].plot(t, filtered, "bo")
    axs[1].plot(t, heights, "ro")
    axs[1].axhline(y=MIN_PEAK_HEIGHT, color="b")
    plt.show()
    # '''

    # narrow down to area of interest
    filtered = [freq for freq in filtered if freq != 0]

    return filtered


def process_tones(freqs, protocol):
    """Convert sampled tones to bitstring."""
    # retrieve list of possible emitted tones based on protocol
    tone_map = encoding.generate_tone_maps(protocol)
    inverse = {v: k for k, v in tone_map.items()}
    possible_tones = list(inverse.keys())

    # quantize the sampled tones by finding the nearest match
    quantized = [find_closest(freq, possible_tones) for freq in freqs]
    # '''
    # View how sampled tones are quantized.
    plt.plot(freqs, "ro")
    plt.plot(quantized, "bo")
    plt.show()
    # '''

    # convert the sampled tones into a frequency list of bits
    sampled_bits = [inverse[freq] for freq in quantized]
    bits_freq_list = condense(sampled_bits)

    # filter frequency list to compile final sampled bitstring
    EXPECTED_NUM_SAMPLES_PER_TONE = 8  # TODO calculate the actual value using protocol and STFT value
    sampled_bitstring = ""
    for bits, num_samples in bits_freq_list:
        num_samples = round(num_samples / EXPECTED_NUM_SAMPLES_PER_TONE)
        sampled_bitstring += bits * num_samples

    return sampled_bitstring


def find_closest(num, sorted_vals):
    closest_val = sorted_vals[0]
    for val in sorted_vals:
        if abs(val - num) < abs(closest_val - num): closest_val = val
        if val > num: break
    return closest_val


def condense(samples):
    """Turn the list of tones into a (freq, number of consecutive samples)."""
    condensed = []
    current_sample = None
    sample_num = 0
    for i in samples:
        if i is not current_sample:
            if current_sample is not None: condensed.append((current_sample, sample_num))
            current_sample = i
            sample_num = 0
        sample_num += 1
    condensed.append((current_sample, sample_num))

    return condensed


def output_txt_file(bitstring, compressed):
    input_string = int(bitstring, 2)
    num_bytes = (input_string.bit_length() + 7) // 8
    byte = input_string.to_bytes(num_bytes, "big")

    if compressed: auxiliary_functions.decompress(byte)

    with open(f"../data/{time.strftime('%m%d-%H%M%S')}.txt", "wb") as f:
        f.write(byte)

    print("File written successfully.")
