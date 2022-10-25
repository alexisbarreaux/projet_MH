from time import time
from typing import Tuple
import os

import numpy as np
from heuristics import descending_random_from_clique
from utils import i_th_nodes_removal_neighbour


class ExploringMetaHeuristicRunner:
    """
     The idea is to start from a given clique and:
    - for a given set of iterations attempt to find a better neighbour
        - if a better one is given restart descending from him
        - otherwise stop this descent
    - for another given set of iterations, explore further from the best know
        - if a better one is found move from there
        - else restart from best known
    """

    # Initialization variables
    graph: np.ndarray
    degrees: np.ndarray
    starting_clique: np.ndarray
    biggest_neighbourhood_size: int
    max_time: int
    verbose: bool
    # Result storing variables
    store_cliques: bool = False
    storing_file: str
    # Runtime variables
    start_time: float
    number_of_iterations: int
    neighbourhood_size: int
    current_biggest_neighbourhood_size: int
    # Current best solution variables
    current_clique: np.ndarray
    best_clique: np.ndarray
    time_best_clique: float
    size_best_clique: int
    iteration_best_clique: int
    # Exploration variables
    exploring_iterations_without_improve: int
    max_best_iterations_without_improve: int
    max_exploring_iterations_without_improve: int

    def __init__(
        self,
        graph: np.ndarray,
        degrees: np.ndarray,
        starting_clique: np.ndarray,
        biggest_neighbourhood_size: int = 5,
        max_time: int = 60,
        verbose: bool = False,
        max_best_iterations_without_improve: int = 5,
        max_exploring_iterations_without_improve: int = 10,
        **kwargs,
    ) -> None:
        """
        Arguments:
            - graph : the graph in which we are working
            - degrees : the degrees in our graph
            - starting_clique : the given starting solution
            - biggest_neighbourhood_size: the max i for V_i our neighbourhood
            - max_time : max_time to run the heuristic in seconds.
            - verbose : wether to print some informations during runtime or not.
            - max_best_iterations_without_improve: used for local descent on best.
            - max_exploring_iterations_without_improve: used for local descent on current.
        """
        self.graph = graph
        self.degrees = degrees
        self.starting_clique = starting_clique
        self.biggest_neighbourhood_size = biggest_neighbourhood_size
        self.max_time = max_time
        self.verbose = verbose
        self.max_best_iterations_without_improve = max_best_iterations_without_improve
        self.max_exploring_iterations_without_improve = (
            max_exploring_iterations_without_improve
        )

    def reset_runtime_variables(self) -> None:
        self.start_time = time()
        self.best_clique = self.starting_clique
        self.current_clique = np.copy(self.best_clique)
        self.size_best_clique = np.sum(self.best_clique)
        self.time_best_found = 0
        self.iteration_best_clique = 0
        self.number_of_iterations = 0
        self.iterations_without_improve = 0
        return

    def get_current_biggest_neighbourhood_size(self, size_of_clique: int) -> int:
        return min(size_of_clique - 1, self.biggest_neighbourhood_size)

    def update_best_solution(self, new_clique: np.ndarray, new_size: int) -> None:
        self.time_best_found = time() - self.start_time
        self.size_best_clique = new_size
        self.best_clique = new_clique
        self.iteration_best_clique = self.number_of_iterations

        if self.store_cliques:
            self.store_best_clique()

        return

    def update_store_variables(
        self, store_cliques: bool = False, storing_file: str = None
    ) -> None:
        self.store_cliques = store_cliques
        if storing_file is None:
            storing_file = "./expl_v2_results.txt"

        # Remove storing file if it already exists
        try:
            os.remove(storing_file)
        except OSError:
            pass
        self.storing_file = storing_file

        # Start by storing starting clique
        self.store_clique(clique=self.starting_clique)

        return

    def store_clique(self, clique: np.ndarray = None) -> None:
        if clique is None:
            clique = self.best_clique
        with open(self.storing_file, "a") as result_file:
            result_file.write(f"{clique}\n\n")
        return

    def print_new_solution(self) -> None:
        print(
            "Elapsed time",
            "{:.2e}".format(self.time_best_found),
            "Best",
            self.size_best_clique,
            "Iteration",
            self.number_of_iterations,
            "Time",
            self.time_best_found,
        )
        return

    def search_locally_from_best_clique(self) -> None:
        """
        Local descent from current best clique for at most max_best_iterations_without_improve.

        This is the intensification part.
        """
        current_biggest_neighbourhood_size = (
            self.get_current_biggest_neighbourhood_size(self.size_best_clique)
        )
        for _ in range(self.max_best_iterations_without_improve):
            self.number_of_iterations += 1

            # Check several neighbourhoods
            for neighbourhood_size in range(1, current_biggest_neighbourhood_size + 1):
                # Remove some nodes from the clique taken randomly
                nodes_to_delete = i_th_nodes_removal_neighbour(
                    self.best_clique, i=neighbourhood_size
                )
                new_clique = np.copy(self.best_clique)
                for node in nodes_to_delete:
                    new_clique[node] = 0
                # Search new best clique greedily from there.
                descending_random_from_clique(
                    clique=new_clique, graph=self.graph, degrees=self.degrees
                )
                # See if the new result is better
                new_size = np.sum(new_clique)
                if new_size > self.size_best_clique:
                    self.update_best_solution(new_clique=new_clique, new_size=new_size)
                    if self.verbose:
                        self.print_new_solution()
                    # If we find a better clique call this again
                    return self.search_locally_from_best_clique()

        return

    def explore_from_current_clique(self) -> None:
        """
        This is the diversification part
        """
        # Search in neighbourhoods
        best_neighbour = None
        size_best_neighbour = 0
        self.number_of_iterations += 1

        for neighbourhood_size in range(1, self.current_biggest_neighbourhood_size + 1):
            new_clique = np.copy(self.current_clique)
            # Remove some nodes from the clique taken randomly
            nodes_to_delete = i_th_nodes_removal_neighbour(
                self.current_clique, i=neighbourhood_size
            )
            for node in nodes_to_delete:
                new_clique[node] = 0
            # Search new best clique greedily from there.
            descending_random_from_clique(
                clique=new_clique, graph=self.graph, degrees=self.degrees
            )
            # See if the new result is better
            new_size = np.sum(new_clique)
            # In the exploration phase, being of same value than best is interesting
            # since we have already searched around the current best, so a new one
            # could yield better results.
            if new_size > self.size_best_clique:
                self.update_best_solution(new_clique=new_clique, new_size=new_size)
                if self.verbose:
                    self.print_new_solution()
                self.exploring_iterations_without_improve = 0
                # Intensification
                self.search_locally_from_best_clique()
                # Update current
                self.current_clique = np.copy(self.best_clique)
                # Go to next loop of the while loop
                break
            elif new_size > size_best_neighbour:
                best_neighbour = new_clique
                size_best_neighbour = new_size
            else:
                continue
        else:
            # This code is to be run only if the loop ends without break.
            # If we didn't break previously, it means we went through without finding a strictly better neighbour,
            # thus take the best out of them
            self.current_clique = best_neighbour
        return

    def meta_heuristic(self) -> Tuple[np.ndarray, float, int, int]:
        """
        When finding a new best clique, try to explore from it for some iterations. If a new best is find restart
        this process.
        Otherwise, start exploring further away for a set number of iterations. If the exploration yields nothing
        after a set number of iterations, start again from best.
        """
        # Setup
        self.reset_runtime_variables()
        # Find optimal from current best found on set number of iterations.
        self.search_locally_from_best_clique()
        self.current_clique = np.copy(self.best_clique)

        self.exploring_iterations_without_improve = 0

        while (time() - self.start_time) < self.max_time:
            self.exploring_iterations_without_improve += 1

            # If exploration doesn't yield results, start again.
            if (
                self.exploring_iterations_without_improve
                > self.max_exploring_iterations_without_improve
            ):
                self.current_clique = np.copy(self.best_clique)
                self.exploring_iterations_without_improve = 0
                self.current_biggest_neighbourhood_size = (
                    self.get_current_biggest_neighbourhood_size(
                        size_of_clique=self.size_best_clique
                    )
                )
            else:
                self.current_biggest_neighbourhood_size = (
                    self.get_current_biggest_neighbourhood_size(
                        size_of_clique=np.sum(self.current_clique)
                    )
                )

            # Exploring part
            self.explore_from_current_clique()

        return (
            self.best_clique,
            self.time_best_found,
            self.iteration_best_clique,
            self.number_of_iterations,
        )
