import numpy as np


def moving_average(data, window_size):
    W = window_size
    if len(data) < W:
        return np.nan
    return sum(filter(None, data[-W:])) / W