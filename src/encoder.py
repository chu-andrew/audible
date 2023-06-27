import bitstring
from dataclasses import dataclass

from protocol import Protocol
from auxiliary_functions import compress


@dataclass
class Encoder:
    data_file: str
    protocol: Protocol
    tone_map: dict

    bitstring: str = None
    tones: list = None

    def encode(self):
        """Convert data file to a sequence of tones."""
        self.bitstring = self._convert_to_bitstring(self.data_file)
        self.tones = self._calculate_tones()
        self._add_indicator_tones()

        if self.protocol.debug:
            print(self.bitstring)
            print("len(bits):", len(self.bitstring))
            print(self.tone_map)

            inverse = {v: k for k, v in self.tone_map.items()}
            for tone in self.tones:
                try:
                    print(f"{inverse[tone]}\t{str(tone).zfill(4)}")
                except KeyError:
                    print(f"indicator\t{str(tone).zfill(4)}")

        return self.tones

    def _convert_to_bitstring(self, data_file) -> str:
        with open(data_file, 'rb') as f:
            data = f.read()
            if self.protocol.compressed:
                data = compress(data)

        return str(bitstring.BitArray(data).bin)

    def _calculate_tones(self) -> list[list[int]]:
        """Create chunks out of bitstring, then convert to array of corresponding tones."""
        # chunk_len evenly splits bitstring
        assert len(self.bitstring) % self.protocol.chunk_len == 0

        tones = []
        for chunk in self._chunk_data(self.bitstring, self.protocol.chunk_len):
            assert chunk in self.tone_map
            tones.append(self.tone_map[chunk])

        return tones

    def _add_indicator_tones(self):
        low, high = self.protocol.f_indicator
        indicator = [low, high, low, high, low]
        self.tones = [*indicator, *self.tones, *indicator]

    @staticmethod
    def _chunk_data(bitstring, chunk_len) -> str:
        for i in range(0, len(bitstring), chunk_len):
            chunk = bitstring[i:i + chunk_len]
            yield chunk
