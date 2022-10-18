# The idea is to start from a given clique and randomly remvoe nodes and attempt to rebuild
# a clique from there.

import numpy as np
from time import time

from utils import i_th_nodes_removal_neighbour
from heuristics import descending_degree_glutonous_heuristic_from_clique


def base_vns_meta_heuristic(
    starting_clique: np.ndarray,
    graph: np.ndarray,
    degrees: np.ndarray,
    biggest_neighbourhood_size: int = 3,
    max_time: int = 60,
) -> list:
    """
    Iteratively deletes random nodes from the clique and attempt to rebuild a better one from there.

    Arguments:
        - starting_clique : the given starting solution
        - graph : the graph in which we are working
        - degrees : the degrees in our graph
        - biggest_neighbourhood_size: the max i for V_i our neighbourhood
        - max_time : max_time to run the heuristic in seconds.
    """
    # Setup
    best_clique = starting_clique
    size_best_clique = np.sum(best_clique)
    start_time = time()
    number_of_iterations = 0

    while (time() - start_time) < max_time:
        neighbourhood_size = 1
        number_of_iterations += 1
        print("Iteration", number_of_iterations)
        while neighbourhood_size < biggest_neighbourhood_size + 1:
            # Remove some nodes from the clique taken randomly
            nodes_to_delete = i_th_nodes_removal_neighbour(
                best_clique, i=neighbourhood_size
            )
            print(neighbourhood_size, nodes_to_delete)
            new_clique = np.copy(best_clique)
            for node in nodes_to_delete:
                new_clique[node] = 0

            # Search new best clique greedily from there.
            descending_degree_glutonous_heuristic_from_clique(
                clique=new_clique, graph=graph, degrees=degrees
            )
            print(new_clique)

            # See if the new result is better
            new_size = np.sum(new_clique)
            if new_size > size_best_clique:
                print("Elapsed time", time() - start_time)
                print("Old", size_best_clique, best_clique)
                print("New", new_size, new_clique)
                size_best_clique = new_size
                best_clique = new_clique
                neighbourhood_size = 1
            else:
                neighbourhood_size += 1

    return best_clique
