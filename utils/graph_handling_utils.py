import numpy as np


def check_if_edge_exists_in_adjacency(
    graph: np.ndarray, first_node: int, second_node: int
) -> bool:
    return graph[first_node][second_node] == 1


def get_degrees_in_adjacency(graph: np.ndarray) -> np.ndarray:
    """Just sum each row to know the degrees in the graph.

    O(n**2)
    """
    return np.sum(graph, axis=1)
