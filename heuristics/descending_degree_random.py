# The idea is to build a clique by using the nodes with the biggest degrees first.
# However, to make it more random, we take the size_of_choice nodes with the biggest degree (if possible)
# each time and take one randomly.
# Cliques are built as the list of nodes first, before being transformed to a 0-1 array
from random import randint

import numpy as np

from utils import (
    order_nodes_in_descending_order,
    check_if_edge_exists_in_adjacency,
    transform_node_clique_to_zero_one,
)


def descending_degree_random_heuristic(
    graph: np.ndarray, degrees: np.ndarray, size_of_choice: int = 5
) -> list:
    """
    Start by reordering the nodes by degrees.
    Then at each step take at most size_of_choice nodes with the biggest degrees, take one randomly, and
    attempt to put it in the clique.

    O(n**2) because we need to check for each node except the first if the possibly all
    previous nodes are neighbours, each of which is done in O(1).
    """
    ordered_nodes = order_nodes_in_descending_order(degrees=degrees)
    candidates = list(ordered_nodes)
    clique = []

    while len(candidates) > 0:
        number_of_nodes_to_take = min(size_of_choice, len(candidates))
        # Choose one randomly
        position_to_take = randint(0, number_of_nodes_to_take - 1)
        # Remove the taken node from candidates
        candidate_node = candidates.pop(position_to_take)

        # If at some point the remaining degrees are less than the number
        # of elements in the clique, no more node can be added, and since they
        # are ordered once one node has a degree which is too small, all the
        # next ones too
        if degrees[candidate_node] < len(clique):
            return transform_node_clique_to_zero_one(len(graph), clique)
        elif np.all(
            [
                check_if_edge_exists_in_adjacency(
                    graph=graph, first_node=candidate_node, second_node=clique_node
                )
                for clique_node in clique
            ]
        ):
            clique.append(candidate_node)

        # Naturally let the loop continue as we have remove one node from the candidates.

    return transform_node_clique_to_zero_one(len(graph), clique)


def multiple_descending_degree_random_heuristic(
    graph: np.ndarray,
    degrees: np.ndarray,
    size_of_choice: int = 5,
    number_of_iterations: int = 10,
    return_all_best: bool = False,
) -> list:
    """
    Run the descending_degree_random_heuristic multiple times and keep best
    occurence.
    """
    # Store the best found result as well as possible other bests
    best_clique = None
    size_best_clique = 0
    all_best_found_cliques = set()

    for _ in range(number_of_iterations):
        clique = descending_degree_random_heuristic(graph, degrees, size_of_choice)
        size_clique = np.sum(clique)
        if size_clique > size_best_clique:
            best_clique = clique
            all_best_found_cliques = set()
        elif return_all_best and size_clique == size_best_clique:
            # This is a set so we won't have duplicates.
            all_best_found_cliques.add(clique)

    if not return_all_best:
        return best_clique
    else:
        return best_clique, all_best_found_cliques
