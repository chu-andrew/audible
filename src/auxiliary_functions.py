import lzma
import bitstring


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


def convert_to_bitstring(filename, protocol) -> str:
    with open(filename, 'rb') as f:
        data = f.read()
        if protocol.compressed:
            data = compress(data)

    return str(bitstring.BitArray(data).bin)


def chunk_data(bitstring, chunk_len):
    for i in range(0, len(bitstring), chunk_len):
        chunk = bitstring[i:i + chunk_len]
        yield chunk


def decode_txt_file(bitstring, compressed):
    """Decode an encoded bitstring back into a text file."""

    input_string = int(bitstring, 2)
    num_bytes = (input_string.bit_length() + 7) // 8
    byte = input_string.to_bytes(num_bytes, "big")

    if compressed:
        return decompress(byte)
    else:
        return byte.decode()  # decode() has option for ignore Unicode errors
