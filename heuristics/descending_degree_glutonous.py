# The idea is to build a clique by using the nodes with the biggest degrees first.
# This is fully deterministic.

import numpy as np

from utils import order_nodes_in_descending_order, check_if_edge_exists_in_adjacency


def descending_degree_glutonous_heuristic(
    graph: np.ndarray, degrees: np.ndarray
) -> list:
    """
    Start by reordering the nodes by degrees, then starting from the one with the
    biggest degree, try to build the biggest possible clique.

    O(n**2) because we need to check for each node except the first if the possibly all
    previous nodes are neighbours, each of which is done in O(1).
    """
    ordered_nodes = order_nodes_in_descending_order(degrees=degrees)

    # Start clique with node of biggest degree
    clique: list = [ordered_nodes[0]]
    # Possible candidates are all other nodes
    candidates = ordered_nodes[1:]
    for candidate_node in candidates:
        if np.all(
            [
                check_if_edge_exists_in_adjacency(
                    graph=graph, first_node=candidate_node, second_node=clique_node
                )
                for clique_node in clique
            ]
        ):
            clique.append(candidate_node)

    return clique
