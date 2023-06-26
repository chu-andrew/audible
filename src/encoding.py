from auxiliary_functions import convert_to_bitstring, chunk_data


def encode(data_file, protocol) -> list[list[float]]:
    """Main encoding function: converts data file to a sequence of tones."""
    bits = convert_to_bitstring(data_file, protocol)
    tone_map = generate_tone_maps(protocol)

    tones = bits_to_tones(bits, tone_map, protocol)

    if protocol.debug:
        print(bits)
        print("len(bits):", len(bits))
        print(tone_map)

        inverse = {v: k for k, v in tone_map.items()}
        for tone in tones:
            print(f"{inverse[tone]}\t{str(tone).zfill(4)}")

    return tones


def generate_tone_maps(protocol) -> dict[str, int]:
    """Map every possible chunk of size chunk_len to a unique tone."""

    f_range = protocol.f_max - protocol.f_min + 1
    possibilities_per_channel = 2 ** protocol.chunk_len

    dF = int(f_range / (possibilities_per_channel * protocol.num_channels - 1))
    assert f_range > possibilities_per_channel * protocol.num_channels
    assert dF >= 1

    prev_f = protocol.f_min
    tone_map = {"0".zfill(protocol.chunk_len): prev_f, }

    for i in range(1, possibilities_per_channel):
        num = bin(i)[2:].zfill(protocol.chunk_len)
        tone_map[num] = prev_f + dF
        prev_f = tone_map[num]

    return tone_map


def bits_to_tones(bitstring, tone_map, protocol) -> list[list[float]]:
    """Create chunks out of bitstring, then convert to array of corresponding tones."""
    # chunk_len evenly splits bitstring
    assert len(bitstring) % protocol.chunk_len == 0

    tones = []
    for chunk in chunk_data(bitstring, protocol.chunk_len):
        assert chunk in tone_map
        tones.append(tone_map[chunk])

    return tones
