import audio
import encoding


def generator(data, chunk_len, debug):
    tones = encoding.get_tones(data, chunk_len, debug)
    audio.play_frequency(tones)


if __name__ == '__main__':
    data_file = "../Data/hello_world.txt"

    chunk_length = 4
    debug_print = True

    generator(data_file, chunk_length, debug_print)
