# The idea is to randomly take the nodes one by one from the
# candidates.
from random import randint

import numpy as np

from utils import (
    check_if_edge_exists_in_adjacency,
)


def descending_random_from_clique(
    clique: np.ndarray, graph: np.ndarray, degrees: np.ndarray
) -> None:
    """
    From a base clique attempt to build a bigger one by regarding the nodes not
    yet in the clique by descending order.
    Updates the base clique in place.
    """
    clique_size = np.sum(clique)
    clique_nodes = [i for i in range(len(clique)) if clique[i] == 1]
    # Take nodes not yet in the clique as candidates and shuffle them
    candidates = [i for i in range(len(clique)) if clique[i] == 0]
    np.random.shuffle(candidates)

    for candidate_node in candidates:
        # A node can't be added if his degree is below clique size.
        if degrees[candidate_node] < clique_size:
            continue
        elif np.all(
            [
                check_if_edge_exists_in_adjacency(
                    graph=graph, first_node=candidate_node, second_node=clique_node
                )
                for clique_node in clique_nodes
            ]
        ):
            clique_nodes.append(candidate_node)
            clique[candidate_node] = 1
            clique_size += 1
        else:
            continue

    return
