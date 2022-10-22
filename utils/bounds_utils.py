import numpy as np
from linearProg.PL import LP


def bound_degrees(degrees: np.ndarray):
    freq = np.bincount(degrees)
    deg = len(freq) - 1
    stop = False
    while deg > 0:
        if freq[deg] > deg:
            return deg + 1
        deg -= 1
    return 1


def bound_relaxed(graph: np.ndarray):
    return LP(graph)
