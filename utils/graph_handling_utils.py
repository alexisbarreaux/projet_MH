import numpy as np


def check_if_edge_exists_in_adjacency(
    graph: np.ndarray, first_node: int, second_node: int
) -> bool:
    return graph[first_node][second_node] == 1


def check_clique_validity(
    graph: np.ndarray,
    degrees: np.ndarray,
    clique: np.ndarray,
    clique_size: int,
    new_node: int,
) -> bool:
    """
    If a new node is valid, then when computing the bit sum of the row of this node
    in the graph and the current clique, clique_size bits should switch from 1 to 0 and
    thus the sum of the resulting array should be degrees[new_node] - clique_size
    """
    return np.sum((graph[new_node] + clique) % 2) == degrees[new_node] - clique_size


def get_degrees_in_adjacency(graph: np.ndarray) -> np.ndarray:
    """Just sum each row to know the degrees in the graph.

    O(n**2)
    """
    return np.sum(graph, axis=1)


def delete_node_from_graph(graph: np.ndarray, node: int) -> None:
    """
    Delete all edges from or to a node.
    """
    graph[node] = np.zeros(len(graph))
    graph[:, node] = np.zeros(len(graph))
    return
