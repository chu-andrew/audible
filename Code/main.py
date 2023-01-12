import audio
import encoding


def generator(data, chunk_len, compress, debug):
    tones = encoding.get_tones(data, chunk_len, compress, debug)
    audio.play_frequency(tones)


if __name__ == '__main__':
    data_file = "../Data/hello_world.txt"

    chunk_length = 4
    compression = False
    debug_print = False

    generator(data_file, chunk_length, compression, debug_print)
