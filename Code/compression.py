import lzma


def compress(bytes_data):
    compressed = lzma.compress(bytes_data)
    print(f"LZMA Compressed: {len(compressed) / len(bytes_data) * 100:.2f}%")

    return compressed


def decompress(bytes_data):
    decompressed = lzma.decompress(bytes_data)
    return decompressed


if __name__ == '__main__':
    with open("../Data/hello_world.txt", "rb") as f:
        print(decompress(compress(f.read())))
