from auxiliary_functions import convert_to_bitstring, chunk_data


def encode(data_file, protocol) -> list[list[float]]:
    """Main encoding function: converts data file to a sequence of tones."""
    bits = convert_to_bitstring(data_file, protocol)
    tone_maps = generate_tone_maps(protocol)

    tones = bits_to_tones(bits, tone_maps, protocol)

    if protocol.debug:
        print(bits)
        for t_map in tone_maps:
            print(t_map)

        inverse = [{v: k for k, v in tone_map.items()} for tone_map in tone_maps]
        for moments in tones:
            for channel_num, tone in enumerate(moments):
                print(f"{inverse[channel_num][tone]}\t{str(tone).zfill(4)}", end='\t\t')
            print()
        print(tones)
    return tones


def generate_tone_maps(protocol) -> list[dict[str, int]]:
    """Map every possible chunk of size chunk_len to a unique tone."""

    f_range = protocol.f_max - protocol.f_min + 1
    possibilities_per_channel = 2 ** protocol.chunk_len

    dF = f_range / (possibilities_per_channel * protocol.num_channels - 1)
    assert f_range > possibilities_per_channel * protocol.num_channels
    assert dF > 1

    maps = []  # holds a tone_map per channel

    prev_f = protocol.f_min
    for _ in range(protocol.num_channels):
        tone_map = {"0".zfill(protocol.chunk_len): prev_f, }

        # TODO use of round() means frequencies are not exactly within bounds
        for i in range(1, possibilities_per_channel):
            num = bin(i)[2:].zfill(protocol.chunk_len)
            tone_map[num] = round(prev_f + dF)
            prev_f = tone_map[num]
        prev_f = round(prev_f + dF)
        maps.append(tone_map)

    return maps


def bits_to_tones(bitstring, tone_map, protocol) -> list[list[float]]:
    """Create chunks out of bitstring, then convert to array of corresponding tones."""
    # chunk_len evenly splits bitstring
    assert len(bitstring) % protocol.chunk_len == 0
    # number of channels evenly splits number of chunks
    assert (len(bitstring) / protocol.chunk_len) % protocol.num_channels == 0

    tones = []  # [[moment1.channel1, moment1.channel2], [moment2.channel1, moment2.channel2], ...]
    for count, chunk in enumerate(chunk_data(bitstring, protocol.chunk_len)):
        channel_num = count % protocol.num_channels  # rotate between channels
        if channel_num == 0: tones.append([])  # when returning to 0th channel, initialize new moment array

        assert chunk in tone_map[channel_num]
        tones[-1].append(tone_map[channel_num][chunk])

    return tones
