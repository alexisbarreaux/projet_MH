import numpy as np
from linearProg.PL import LP


def bound_degrees(degrees: np.ndarray):
    freq = np.bincount(degrees)
    deg = len(freq) - 1
    somme = 0
    while deg > 0:
        somme += freq[deg]
        if somme > deg:
            return deg + 1
        deg -= 1
    return 1


def bound_relaxed(graph: np.ndarray):
    return LP(graph)
