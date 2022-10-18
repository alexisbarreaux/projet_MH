import numpy as np


def order_nodes_in_descending_order(degrees: np.ndarray) -> np.ndarray:
    """
    Argsorting the opposite of degrees to get nodes
    in descending degrees order.
    """
    return np.argsort(-1 * degrees)


def order_subset_of_nodes_in_descending_order(
    nodes: np.ndarray, degrees: np.ndarray
) -> np.ndarray:
    """
    Argsorting the opposite of degrees to get nodes
    in descending degrees order. We only order a subset
    of the nodes.
    """
    sub_degrees = np.array([-1 * degrees[node] for node in nodes])
    sorted_degrees = np.argsort(sub_degrees)

    return np.array([nodes[i] for i in sorted_degrees])
