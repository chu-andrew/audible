import encoding


def decode_txt_file(bitstring):
    input_string = int(bitstring, 2)
    num_bytes = (input_string.bit_length() + 7) // 8
    input_array = input_string.to_bytes(num_bytes, "big")

    ascii_data = input_array.decode()
    return ascii_data


# look at anchor project for receiver


if __name__ == "__main__":
    data_file = "../Data/hello_world.txt"
    bits = encoding.get_bitstring(data_file)
    print(bits)

    decoded = decode_txt_file(bits)
    print(decoded)
