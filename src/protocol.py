from dataclasses import dataclass

import numpy as np
import pyaudio
import yaml


@dataclass
class Protocol:
    num_channels: int
    sample_rate: int
    chunk_len: int
    recording_seconds: float

    f_min: float
    f_max: float
    f_indicator: tuple

    np_format: np.dtype
    pa_format: int

    moment_len: float  # defines how long each tone will be broadcast for
    volume: float  # [0.0, 1.0], used for pyaudio output

    compressed: bool
    debug: bool

    def __init__(self, config="../data/config.yaml"):
        self.np_format = np.dtype(np.single)
        self.pa_format = pyaudio.paFloat32

        with open(config, "r") as f:
            cfg = yaml.full_load(f)

            self.num_channels = cfg["num_channels"]
            self.sample_rate = cfg["sample_rate"]
            self.chunk_len = cfg["chunk_len"]
            self.recording_seconds = cfg["recording_seconds"]

            self.f_min = cfg["f_min"]
            self.f_max = cfg["f_max"]
            self.f_indicator = tuple(cfg["f_indicator"])

            self.moment_len = cfg["moment_len"]
            self.volume = cfg["volume"]

            self.compressed = cfg["compressed"]
            self.debug = cfg["debug"]
