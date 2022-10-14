import numpy as np


def order_nodes_in_descending_order(degrees: np.ndarray) -> np.ndarray:
    """
    Argsorting the opposite of degrees to get nodes
    in descending degrees order.
    """
    return np.argsort(-1 * degrees)
