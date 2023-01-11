import bitstring


def get_bitstring(filename):
    return str(bitstring.BitArray(filename=filename).bin)


def generate_tone_map(chunk_len):
    F_MIN = 200  # 1875.000  # Hz
    F_MAX = 6000  # 6328.125

    f_range = F_MAX - F_MIN
    possibilities = 2 ** chunk_len

    dF = f_range / (possibilities - 1)
    assert f_range > possibilities
    assert dF > 1

    tone_map = {"0".zfill(chunk_len): F_MIN, }
    prev_f = F_MIN

    for i in range(1, possibilities):
        num = bin(i)[2:].zfill(chunk_len)
        tone_map[num] = round(prev_f + dF)
        prev_f = tone_map[num]

    return tone_map


def bits_to_tones(data, tone_map, chunk_len):
    tones = []
    assert len(data) % chunk_len == 0

    for chunk in chunk_data(data, chunk_len):
        assert chunk in tone_map
        tones.append(tone_map[chunk])
    return tones


def chunk_data(data, chunk_len):
    # TODO bits_to_tones() and chunk_data() can probably be made into a one-liner with list comprehension later
    for i in range(0, len(data), chunk_len):
        chunk = data[i:i + chunk_len]
        yield chunk


def get_tones(data, chunk_len, debug):
    bits = get_bitstring(data)
    tone_map = generate_tone_map(chunk_len)

    tones = bits_to_tones(bits, tone_map, chunk_len)

    if debug:
        print(bits)
        print(tone_map)

        inverse_tone_map = {v: k for k, v in tone_map.items()}
        for t in tones:
            print(f"{inverse_tone_map[t]}\t{str(t).zfill(4)}")

    return tones


if __name__ == '__main__':
    data_file = "../Data/hello_world.txt"
    tones = get_tones(data_file, 2, True)
