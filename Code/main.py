import audio
import encoding
import fourier


def generator(data, chunk_len, num_channels, compress, debug):
    tones = encoding.get_tones(data, chunk_len, num_channels, compress, debug)
    audio.play_frequency(tones)



if __name__ == '__main__':
    data_file = "../Data/hello_world.txt"

    chunk_length = 2
    channels = 2
    compression = False
    debug_print = False

    fourier.receive(14,44100,1024, 'test.wav'); generator(data_file, chunk_length, channels, compression, debug_print)


