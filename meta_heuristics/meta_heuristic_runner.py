from abc import abstractclassmethod
from time import time
from typing import Tuple

import numpy as np
from heuristics import descending_random_from_clique
from utils import i_th_nodes_removal_neighbour

from meta_heuristic_runner import BaseMetaHeuristicRunner

# UNUSED


class BaseMetaHeuristicRunner:
    """
    Base abstract class to define our meta heuristics
    """

    # Initialization variables
    graph: np.ndarray
    degrees: np.ndarray
    starting_clique: np.ndarray
    biggest_neighbourhood_size: int
    verbose: bool
    max_time: int
    # Runtime variables
    start_time: float
    number_of_iterations: int
    neighbourhood_size: int
    current_biggest_neighbourhood_size: int
    # Current best solution variables
    current_best_clique: np.ndarray
    time_best_clique: float
    size_best_clique: int
    iteration_best_clique: int
    # New evaluated solution
    new_size: int = 0

    def __init__(
        self,
        graph: np.ndarray,
        degrees: np.ndarray,
        starting_clique: np.ndarray,
        biggest_neighbourhood_size: int = 3,
        max_time: int = 60,
        verbose: bool = False,
    ) -> None:
        """
        Arguments:
            - graph : the graph in which we are working
            - degrees : the degrees in our graph
            - starting_clique : the given starting solution
            - biggest_neighbourhood_size: the max i for V_i our neighbourhood
            - max_time : max_time to run the heuristic in seconds.
            - verbose : wether to print some informations during runtime or not.
        """
        self.graph = graph
        self.degrees = degrees
        self.starting_clique = starting_clique
        self.biggest_neighbourhood_size = biggest_neighbourhood_size
        self.max_time = max_time
        self.verbose = verbose

    @abstractclassmethod
    def print_amelioration_information() -> None:
        return

    def set_or_reset_runtime_variables(self) -> None:
        self.current_best_clique = self.starting_clique
        self.size_best_clique = np.sum(self.starting_clique)
        self.start_time = time()
        self.time_best_clique = 0
        self.number_of_iterations = 0
        self.iteration_best_clique = 0

    def stopping_condition(self) -> bool:
        return (time() - self.start_time) < self.max_time

    @abstractclassmethod
    def meta_heuristic_iteration() -> None:
        """
        Runs one step from current best clique.
        """
        return

    def run_meta_heuristic(
        self,
    ) -> Tuple[np.ndarray, float, int, int]:
        # Setup
        self.set_or_reset_runtime_variables()

        while not self.stopping_condition():
            self.number_of_iterations += 1
            self.meta_heuristic_iteration()

        return (
            self.current_best_clique,
            self.time_best_clique,
            self.iteration_best_clique,
            self.number_of_iterations,
        )


class VNS_BaseMetaHeuristicRunner(BaseMetaHeuristicRunner):
    def set_current_biggest_neighbourhood(self) -> None:
        self.current_biggest_neighbourhood_size = min(
            self.size_best_clique, self.biggest_neighbourhood_size
        )

    def print_amelioration_information(self) -> None:
        print(
            "Elapsed time",
            self.time_best_clique,
            "Old",
            self.size_best_clique,
            "New",
            self.new_size,
            "Iteration",
            self.number_of_iterations,
            "Neighbourhood size",
            self.neighbourhood_size,
        )
        return

    def check_if_new_clique_is_better(self, new_clique: np.ndarray) -> None:
        new_size = np.sum(new_clique)
        # If it is better store and return to V_1
        if new_size > self.size_best_clique:
            self.time_best_clique = time() - self.start_time
            if self.verbose:
                self.print_amelioration_information()
            self.size_best_clique = new_size
            self.current_best_clique = new_clique
            self.neighbourhood_size = 1
            self.iteration_best_clique = self.number_of_iterations
        # Else go to V_i+1
        else:
            self.neighbourhood_size += 1

    def meta_heuristic_iteration(
        self,
    ) -> None:
        """
        Iteratively deletes random nodes from the clique and attempt to rebuild a better one from there.
        """
        # We can't take a neighbourhood bigger than all the nodes in the clique.
        self.set_current_biggest_neighbourhood()

        self.neighbourhood_size = 1
        while self.neighbourhood_size < self.current_biggest_neighbourhood_size + 1:
            # Remove some nodes from the clique taken randomly
            nodes_to_delete = i_th_nodes_removal_neighbour(
                self.current_best_clique, i=self.neighbourhood_size
            )
            new_clique = np.copy(self.current_best_clique)
            for node in nodes_to_delete:
                new_clique[node] = 0

            # Search new best clique greedily from there.
            descending_random_from_clique(
                clique=new_clique, graph=self.graph, degrees=self.degrees
            )

            # See if the new result is better
            self.check_if_new_clique_is_better()
