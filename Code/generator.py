import audio
import encoding


if __name__ == '__main__':
    data_file = "../Data/hello_world.txt"
    tones = encoding.get_tones(data_file, 4)

    audio.play_frequency(tones, [0.5 for _ in range(len(tones))])
