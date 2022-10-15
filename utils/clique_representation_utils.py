import numpy as np


def transform_node_clique_to_zero_one(number_of_nodes: int, clique: list) -> np.ndarray:
    clique_zero_one = np.zeros(number_of_nodes, dtype=np.int8)
    for node in clique:
        clique_zero_one[node] = 1
    return clique_zero_one


def transform_zero_one_clique_to_nodes(clique: np.ndarray) -> list:
    clique_nodes = []
    for i in range(len(clique)):
        if clique[i] == 1:
            clique_nodes.append(i)
    return clique_nodes
