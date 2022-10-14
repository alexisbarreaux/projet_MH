import itertools

import numpy as np


def bit_flip(array: np.ndarray, positions: list[int]) -> np.ndarray:
    """Takes a binary array and switches the bits at the wanted positions."""
    copied_array = np.copy(array)
    for position in positions:
        copied_array[position] = 1 - copied_array[position]
    return copied_array


def i_th_bit_flip_neighbourhood(array: np.ndarray, i: int) -> list:
    """
    Creates the i-th neighbourhood of an array x, meaning V_i(x)
    """
    positions = np.arange(len(array))
    positions_combinations = itertools.combinations(positions, i)
    return [
        bit_flip(array, positions_combination)
        for positions_combination in positions_combinations
    ]
