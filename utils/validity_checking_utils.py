import numpy as np


def check_validity_for_adjacency(graph: np.ndarray, clique: list):
    """
    Function to check if a clique is valid. It runs in O(n**2) and is not
    meant to be efficient.
    """
    n = len(clique)
    adjacency_values = []
    for i in range(n):
        for j in range(i + 1, n):
            first_node, second_node = clique[i], clique[j]
            adjacency_values.append(graph[first_node, second_node])

    return all(adjacency_values)
