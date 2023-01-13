import encoding
import compression


def decode_txt_file(bitstring, compressed):
    input_string = int(bitstring, 2)
    num_bytes = (input_string.bit_length() + 7) // 8
    byte = input_string.to_bytes(num_bytes, "big")

    if compressed:  return compression.decompress(byte)
    else:           return byte.decode()  # decode() has option for ignore Unicode errors


if __name__ == "__main__":
    compress = True
    #data_file = "../Data/hello_world.txt"
    data_file = "../Data/hello_world.txt"

    encoded = encoding.get_bitstring(data_file, compress)
    print(encoded)

    decoded = decode_txt_file(encoded, compress)
    print(decoded)


