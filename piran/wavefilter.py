import numpy as np


class WaveFilter:

    def filter(self, k: np.ndarray):
        if k.size == 0:
            return np.nan
        elif k.size == 1:
            return k[0]
        else:
            msg = "We got more than one real positive root for k"
            raise ValueError(msg)
