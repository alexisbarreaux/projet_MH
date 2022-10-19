# The idea is to start from a given clique and randomly remvoe nodes and attempt to rebuild
# a clique from there.
from typing import Tuple

import numpy as np
from time import time
from random import random

from utils import i_th_nodes_removal_neighbour
from heuristics import descending_random_from_clique


def enhansed_base_vns_meta_heuristic(
    starting_clique: np.ndarray,
    graph: np.ndarray,
    degrees: np.ndarray,
    biggest_neighbourhood_size: int = 3,
    max_iterations_without_improvement: int = 10,
    max_time: int = 60,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Iteratively deletes random nodes from the clique and attempt to rebuild a better one from there.
    If no improvement is made after a set number of iterations, restart from a random node.

    Arguments:
        - starting_clique : the given starting solution
        - graph : the graph in which we are working
        - degrees : the degrees in our graph
        - biggest_neighbourhood_size: the max i for V_i our neighbourhood
        - max_iterations_without_improvement: maximum iterations without any improvement.
        - max_time : max_time to run the heuristic in seconds.
        - verbose : wether to print some informations during runtime or not.
    """
    # Setup
    start_time = time()
    time_best_found = 0
    number_of_iterations = 0
    iteration_of_best = 0
    # We can't take a neighbourhood bigger than all the nodes in the clique.
    # We make the choice of removing at most all the nodes but one.
    current_biggest_neighbourhood_size = min(
        np.sum(starting_clique) - 1, biggest_neighbourhood_size
    )
    # Best found variables
    best_clique = starting_clique
    size_best_clique = np.sum(best_clique)
    # Current values valuables
    current_clique = starting_clique

    # Also store how many times we found ourselves bloked on each node in a clique
    iterations_without_improvement = 0
    times_blocked_on_nodes = np.ones(len(graph))

    def reset_to_one_node_clique(times_blocked_on_nodes: np.ndarray) -> np.ndarray:
        """
        Function to reset to a single node clique if to many iterations go
        by without improvement.
        The new node to start from is taken with a probability higher if we have
        been blocked less times in it
        """
        times_blocked_on_nodes = times_blocked_on_nodes + best_clique
        # Cumsum the values and then inverse them to get probabilities intervals
        probabilities_intervals = np.cumsum(1 / times_blocked_on_nodes)
        # Take a random value between 0 and the max of the intervals
        random_value = random() * probabilities_intervals[-1]
        # Find the linked node
        node = np.searchsorted(probabilities_intervals, random_value)
        # Build the clique
        new_clique = np.zeros(len(best_clique))
        new_clique[node] = 1
        return new_clique

    while (time() - start_time) < max_time:
        # Avoid excessive number of iterations without improvement
        iterations_without_improvement += 1
        if iterations_without_improvement > max_iterations_without_improvement:
            iterations_without_improvement = 0
            # Reset to a single node
            current_clique = reset_to_one_node_clique(times_blocked_on_nodes)
            # Descend from there
            descending_random_from_clique(
                clique=current_clique, graph=graph, degrees=degrees
            )
            current_biggest_neighbourhood_size = min(
                np.sum(current_clique) - 1, biggest_neighbourhood_size
            )

        number_of_iterations += 1
        neighbourhood_size = 1

        while neighbourhood_size < current_biggest_neighbourhood_size + 1:
            # Remove some nodes from the clique taken randomly
            nodes_to_delete = i_th_nodes_removal_neighbour(
                best_clique, i=neighbourhood_size
            )
            new_clique = np.copy(current_clique)
            for node in nodes_to_delete:
                new_clique[node] = 0

            # Search new best clique greedily from there.
            descending_random_from_clique(
                clique=new_clique, graph=graph, degrees=degrees
            )

            # See if the new result is better
            new_size = np.sum(new_clique)
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
                best_clique = current_clique = new_clique
                neighbourhood_size = 1
                iteration_of_best = number_of_iterations
                iterations_without_improvement = 0
                # Update neighbourhood_size
                current_biggest_neighbourhood_size = min(
                    np.sum(current_clique) - 1, biggest_neighbourhood_size
                )
            else:
                neighbourhood_size += 1

    return best_clique, time_best_found, iteration_of_best, number_of_iterations
