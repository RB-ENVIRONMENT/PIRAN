class GaussParams:
    def __init__(self, range, peak, width):
        self._range = range
        self._peak = peak
        self._width = width


class Waves:
    def __init__(self, normal_angles: GaussParams, frequencies: GaussParams):
        self.angles = normal_angles
        self.frequencies = frequencies
