import itertools

import numpy as np
import random


def bit_flip(array: np.ndarray, positions: list[int]) -> np.ndarray:
    """Takes a binary array and switches the bits at the wanted positions."""
    copied_array = np.copy(array)
    for position in positions:
        copied_array[position] = 1 - copied_array[position]
    return copied_array


def i_th_bit_flip_neighbourhood(array: np.ndarray, i: int) -> list:
    """
    Creates the i-th neighbourhood of an array x, meaning V_i(x)
    where the transformation is to flip a bit.
    """
    positions = np.arange(len(array))
    positions_combinations = itertools.combinations(positions, i)
    return [
        bit_flip(array, positions_combination)
        for positions_combination in positions_combinations
    ]


def i_th_nodes_removal_neighbour(array: np.ndarray, i: int) -> list:
    """
    Creates a i-th neighbour of an array x, meaning x' in V_i(x).
    Here the transformation is to remove at most i bits to 1 in the array,
    meaning removing i-nodes.
    The result returned is not the arrays but the separate combinations of
    nodes one can remove.
    """
    clique_nodes = [i for i in range(len(array)) if array[i] == 1]
    return random.sample(clique_nodes, i)
