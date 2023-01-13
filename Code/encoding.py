import bitstring
import compression


def get_bitstring(filename, compress):
    with open(filename, 'rb') as f:
        data = f.read()
    if compress: data = compression.compress(data)
    return str(bitstring.BitArray(data).bin)


def generate_tone_maps(chunk_len, channels):
    F_MIN = 200  # 1875.000  # Hz
    F_MAX = 5000  # 6328.125

    f_range = F_MAX - F_MIN
    possibilities_per_channel = 2 ** chunk_len

    dF = f_range / (possibilities_per_channel * channels - 1)
    assert f_range > possibilities_per_channel * channels
    assert dF > 1

    maps = []

    prev_f = F_MIN
    for c in range(channels):
        tone_map = {"0".zfill(chunk_len): prev_f, }

        # TODO use of round() means frequencies are not exactly within bounds
        for i in range(1, possibilities_per_channel):
            num = bin(i)[2:].zfill(chunk_len)
            tone_map[num] = round(prev_f + dF)
            prev_f = tone_map[num]
        prev_f = round(prev_f + dF)
        maps.append(tone_map)

    return maps


def bits_to_tones(data, tone_map, chunk_len, channels):
    assert len(data) % chunk_len == 0
    assert len(data) / chunk_len % channels == 0

    tones = []
    for count, chunk in enumerate(chunk_data(data, chunk_len)):
        channel_num = count % channels
        if channel_num == 0: tones.append([])

        assert chunk in tone_map[channel_num]
        tones[-1].append(tone_map[channel_num][chunk])

    return tones


def chunk_data(data, chunk_len):
    for i in range(0, len(data), chunk_len):
        chunk = data[i:i + chunk_len]
        yield chunk


def get_tones(data, chunk_len, channels, compress, debug):
    bits = get_bitstring(data, compress)
    tone_maps = generate_tone_maps(chunk_len, channels)

    tones = bits_to_tones(bits, tone_maps, chunk_len, channels)

    if debug:
        print(bits)
        for t_map in tone_maps: print(t_map)

        inverse_tone_maps = [{v: k for k, v in tone_map.items()} for tone_map in tone_maps]
        for moments in tones:
            for channel_num, tone in enumerate(moments):
                print(f"{inverse_tone_maps[channel_num][tone]}\t{str(tone).zfill(4)}", end='\t\t')
            print()

    return tones
