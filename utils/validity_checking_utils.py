import numpy as np


def check_validity_for_adjacency(graph: np.ndarray, clique: list):
    """
    Function to check if a clique is valid. It runs in O(n**2).
    """
    nodes_in_clique = [i for i in range(len(clique)) if clique[i] == 1]
    adjacency_values = []
    for i in range(len(nodes_in_clique)):
        for j in range(i + 1, len(nodes_in_clique)):
            first_node, second_node = nodes_in_clique[i], nodes_in_clique[j]
            adjacency_values.append(graph[first_node, second_node])

    return all(adjacency_values)
