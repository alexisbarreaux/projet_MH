# The idea is to start from a given clique and randomly remvoe nodes and attempt to rebuild
# a clique from there.
from typing import Tuple

import numpy as np
from time import time

from utils import i_th_nodes_removal_neighbour
from heuristics import descending_random_from_clique


def exploring_base_vns_meta_heuristic(
    starting_clique: np.ndarray,
    graph: np.ndarray,
    degrees: np.ndarray,
    biggest_neighbourhood_size: int = 3,
    max_time: int = 60,
    verbose: bool = False,
    max_iterations_without_improve: int = 5,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Iteratively deletes random nodes from the clique and attempt to rebuild a better one from there.

    Arguments:
        - starting_clique : the given starting solution
        - graph : the graph in which we are working
        - degrees : the degrees in our graph
        - biggest_neighbourhood_size: the max i for V_i our neighbourhood
        - max_time : max_time to run the heuristic in seconds.
        - verbose : wether to print some informations during runtime or not.
        - max_iterations_without_improve: int that defines after how many descents without improve we reset
                                    to the best known clique
    """
    # Setup
    start_time = time()
    best_clique = starting_clique
    current_clique = np.copy(best_clique)
    size_best_clique = np.sum(best_clique)
    time_best_found = 0
    number_of_iterations = 0
    iterations_without_improve = 0
    iteration_of_best = 0

    # We can't take a neighbourhood bigger than all the nodes in the clique.
    current_biggest_neighbourhood_size = min(
        size_best_clique - 1, biggest_neighbourhood_size
    )
    while (time() - start_time) < max_time:
        # Increase iterations counters
        number_of_iterations += 1
        iterations_without_improve += 1
        if iterations_without_improve > max_iterations_without_improve:
            current_clique = np.copy(best_clique)
            current_biggest_neighbourhood_size = min(
                size_best_clique - 1, biggest_neighbourhood_size
            )
        else:
            current_biggest_neighbourhood_size = min(
                np.sum(current_clique) - 1, biggest_neighbourhood_size
            )

        # Restart from first neighbourhood
        neighbourhood_size = 1
        while neighbourhood_size < current_biggest_neighbourhood_size + 1:
            # Remove some nodes from the clique taken randomly
            nodes_to_delete = i_th_nodes_removal_neighbour(
                current_clique, i=neighbourhood_size
            )
            for node in nodes_to_delete:
                current_clique[node] = 0
            # Search new best clique greedily from there.
            descending_random_from_clique(
                clique=current_clique, graph=graph, degrees=degrees
            )
            # See if the new result is better
            new_size = np.sum(current_clique)
            if new_size > size_best_clique:
                time_best_found = time() - start_time
                if verbose:
                    print(
                        "Elapsed time",
                        time_best_found,
                        "Old",
                        size_best_clique,
                        "New",
                        new_size,
                        "Iteration",
                        number_of_iterations,
                        "Neighbourhood size",
                        neighbourhood_size,
                    )
                size_best_clique = new_size
                best_clique = current_clique
                current_clique = np.copy(best_clique)
                neighbourhood_size = 1
                iteration_of_best = number_of_iterations
                iterations_without_improve = 0
            else:
                neighbourhood_size += 1

    return best_clique, time_best_found, iteration_of_best, number_of_iterations
