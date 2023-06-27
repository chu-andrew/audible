import lzma


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


def compress(bytes_data) -> bytes:
    compressed = lzma.compress(bytes_data)
    '''
    # TODO it would be ideal to avoid compression if size is increased. is there a way to encode whether compression has been applied to the output tones?
    if len(compressed) >= len(bytes_data):
        print("LZMA compression increased the size of the data. Cancelling compression.")
        return bytes_data
    else:
        print(f"LZMA compressed: {(1 - len(compressed) / len(bytes_data)) * 100:.2f}%")
        return compressed
    '''
    print(f"LZMA compressed: {(1 - len(compressed) / len(bytes_data)) * 100:.2f}%")
    return compressed


def decompress(bytes_data) -> bytes:
    decompressed = lzma.decompress(bytes_data)
    return decompressed
