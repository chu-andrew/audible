import bitstring


def generate_tone_map(chunk_len):
    Fmin = 1875.000  # Hz
    Fmax = 6328.125

    possibilities = 2 ** chunk_len
    dF = (Fmax - Fmin) / (possibilities - 1)

    tone_map = {"0".zfill(chunk_len): Fmin, }
    prevF = Fmin

    for i in range(1, possibilities):
        num = bin(i)[2:].zfill(chunk_len)
        tone_map[num] = prevF + dF
        prevF = tone_map[num]

    return tone_map


def get_bitstring(filename):
    return str(bitstring.BitArray(filename=filename).bin)


def bits_to_tones(data, tone_map, chunk_len):
    tones = []
    for chunk in chunk_data(data, chunk_len):
        assert chunk in tone_map
        tones.append(tone_map[chunk])
    return tones


def chunk_data(data, chunk_len):
    assert len(data) % chunk_len == 0
    chunks = []
    for i in range(0, len(data), chunk_len):
        chunks.append(data[i:i + chunk_len])
    return chunks


def get_tones(data, chunk_len):
    tone_map = generate_tone_map(chunk_len)
    bits = get_bitstring(data)
    tones = bits_to_tones(bits, tone_map, chunk_len)

    return tones


if __name__ == "__main__":
    data_file = "../Data/hello_world.txt"
    print(get_tones(data_file, 1))
