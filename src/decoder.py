import numpy as np
import scipy
import time
from matplotlib import pyplot as plt
from dataclasses import dataclass
import math

from protocol import Protocol
import auxiliary_functions
from auxiliary_functions import generate_tone_maps


@dataclass
class Decoder:
    protocol: Protocol
    tone_map: dict = None

    # TODO use lists instead of np.array for faster appends? or allocate space ahead of time
    frames_buffer = np.array([])
    max_buffer_size: int = 0
    indicator_window_size: int = 0
    window_checked = False

    session_in_progress: bool = False
    session_frames = np.array([])

    def __post_init__(self):
        self.tone_map = generate_tone_maps(self.protocol)
        self.frames_buffer = np.array([])

        indicator_len = 5 * self.protocol.moment_len * self.protocol.sample_rate
        self.indicator_window_size = math.ceil(2 * indicator_len)
        self.max_buffer_size = math.ceil(3 * indicator_len)

    def write_to_buffer(self, bytes=None, flush_buffer=False):
        if bytes:
            frames = np.frombuffer(bytes, dtype=self.protocol.np_format)
            self.frames_buffer = np.append(self.frames_buffer, frames)

            if self.session_in_progress:
                self.session_frames = np.append(self.session_frames, frames)

        self.check_buffer(flush_buffer)

    def check_buffer(self, flush_buffer):
        if flush_buffer or \
                (not self.window_checked and len(self.frames_buffer) >= self.indicator_window_size):
            self.process_buffer()
            self.window_checked = True

        if len(self.frames_buffer) > self.max_buffer_size:
            self.frames_buffer = self.frames_buffer[-self.indicator_window_size:]
            self.window_checked = False

    def process_buffer(self):
        detected, bitstring = self.decode_section(self.frames_buffer)
        print(bitstring)

        if detected:
            if self.session_in_progress:  # end
                self.process_session()
                self.session_in_progress = False
            else:  # start
                self.session_frames = np.append(self.session_frames, self.frames_buffer)
                self.session_in_progress = True

    def process_session(self):
        _, bitstring = self.decode_section(self.session_frames, True)
        print(bitstring)

        # remove front indicator
        ind = "indicator_aindicator_bindicator_aindicator_bindicator_a"
        l_ind = len(ind)

        bitstring = bitstring[bitstring.index(ind) + l_ind:]
        print(bitstring)

        bitstring = bitstring[:bitstring.index(ind)]
        print(bitstring)

        self.output_txt_file(bitstring, False)

    def decode_section(self, signal, debug=False):
        fourier_data = self.short_time_fourier(signal, debug)
        peaks = self.filter_peaks(*fourier_data, debug)
        sampled_bitstring = self.frequencies_to_bits(peaks, debug)
        if "indicator_aindicator_bindicator_aindicator_bindicator_a" in sampled_bitstring:
            return True, sampled_bitstring
        else:
            return False, sampled_bitstring

    def short_time_fourier(self, signal, debug=False):
        """Run STFT on entirety of captured signal."""
        bits_per_second = self.protocol.sample_rate / self.protocol.chunk_len
        segment_size = int(bits_per_second * self.protocol.moment_len)

        f, t, Zxx = scipy.signal.stft(signal, self.protocol.sample_rate, nperseg=segment_size) # TODO change nperseg
        Zxx = np.abs(Zxx)

        if debug:
            plt.pcolormesh(t, f, Zxx, shading="gouraud")
            plt.title("STFT Magnitude")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")
            plt.show()

        # extract relevant information from STFT (dominant freq and peak height)
        Zxx = np.transpose(Zxx)  # flip rows/cols to get each sample's fft in each row

        frequencies = []
        heights = []
        for sample_stft in Zxx:
            peaks, properties = scipy.signal.find_peaks(
                sample_stft, height=(None, None))  # heights parameter necessary to get properties

            if len(sample_stft[peaks]) > 0:
                max_peak_index = np.argmax(sample_stft[peaks])  # Find the index of the maximum peak within peak array

                freq = f[peaks[max_peak_index]]
                max_height = properties["peak_heights"][max_peak_index]
            else:
                freq, max_height = 0, 0

            frequencies.append(freq)
            heights.append(max_height)

        return frequencies, heights, t

    @staticmethod
    def filter_peaks(frequencies, heights, t, debug=False) -> list[float]:
        """Remove all peaks of insufficient height (strength of signal) and return area of interest."""
        # FIXME: JJ [input: frequencies, heights; output: freqs from area of interest]
        # TODO this is a naive method that may lead to 0s within data area
        MIN_PEAK_HEIGHT = 0.001  # determined with stft heights plot # TODO is there a way to not use a constant?
        filtered = frequencies.copy()
        for i in range(len(filtered)):
            if heights[i] < MIN_PEAK_HEIGHT:
                filtered[i] = 0

        if debug:
            # View all sampled tones, the tones with fft peaks above threshold, and fft peak heights.
            fig, axs = plt.subplots(2)
            axs[0].plot(t, frequencies, "go")
            axs[0].plot(t, filtered, "bo")
            axs[1].plot(t, heights, "ro")
            axs[1].axhline(y=MIN_PEAK_HEIGHT, color="b")
            plt.show()

        # narrow down to area of interest
        filtered = [freq for freq in filtered if freq != 0]

        return filtered

    def frequencies_to_bits(self, freqs, debug=False) -> str:
        """Convert sampled tones to bitstring."""
        # retrieve list of possible emitted tones based on protocol
        inverse = {v: k for k, v in self.tone_map.items()}
        possible_tones = list(inverse.keys())
        possible_tones.insert(0, 0)  # may help to weed out bad values??? not completely sure #FIXME

        # quantize the sampled tones by finding the nearest match
        quantized = [self._find_closest(freq, possible_tones) for freq in freqs]

        if debug:
            # View how sampled tones are quantized.
            plt.plot(freqs, "ro")
            plt.plot(quantized, "bo")
            plt.show()

        # convert the sampled tones into a frequency list of bits
        sampled_bits = [inverse[freq] for freq in quantized]
        bits_freq_list = self._condense(sampled_bits)

        # filter frequency list to compile final sampled bitstring
        EXPECTED_NUM_SAMPLES_PER_TONE = 8  # TODO calculate the actual value using protocol and STFT value
        sampled_bitstring = ""
        for bits, num_samples in bits_freq_list:
            num_samples = round(num_samples / EXPECTED_NUM_SAMPLES_PER_TONE)
            sampled_bitstring += bits * num_samples

        return sampled_bitstring

    @staticmethod
    def _find_closest(num, sorted_vals) -> int:
        closest_val = sorted_vals[0]
        for val in sorted_vals:
            if abs(val - num) < abs(closest_val - num): closest_val = val
            if val > num: break
        return closest_val

    @staticmethod
    def _condense(samples):
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

    @staticmethod
    def output_txt_file(bitstring, compressed) -> None:
        input_string = int(bitstring, 2)
        num_bytes = (input_string.bit_length() + 7) // 8
        byte = input_string.to_bytes(num_bytes, "big")

        if compressed: auxiliary_functions.decompress(byte)

        with open(f"../data/{time.strftime('%m%d-%H%M%S')}.txt", "wb") as f:
            f.write(byte)

        print("File written successfully.")
