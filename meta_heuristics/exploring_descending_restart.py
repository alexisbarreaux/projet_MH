from time import time
from typing import Tuple
from queue import Queue
from random import randint

import numpy as np
from heuristics import descending_random_from_clique_with_taboos
from utils import i_th_nodes_removal_neighbour


class ExploringTaboosRestartMetaHeuristicRunner:
    """
     The idea is to start from a given clique and:
    - for a given set of iterations attempt to find a better neighbour
        - if a better one is given restart descending from him
        - otherwise stop this descent
    - for another given set of iterations, explore further from the best know
        - if a better one is found move from there
        - else restart from best known
    - If after some restarts we still don't get anything better, restart from a random node.
    """

    # Initialization variables
    graph: np.ndarray
    degrees: np.ndarray
    starting_clique: np.ndarray
    biggest_neighbourhood_exploration: int
    verbose: bool
    max_time: int
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
    # Intensification variables
    biggest_neighbourhood_intensification: int
    # Exploration variables
    exploring_iterations_without_improve: int
    restarts_iterations_without_improve: int
    max_intensification_without_improve: int
    max_exploring_iterations_without_improve: int
    max_restarts_iterations_without_improve: int
    # Taboos variables
    taboos_list: np.ndarray
    taboos_queue: Queue
    taboos_queue_size: int
    max_taboos: int

    def __init__(
        self,
        graph: np.ndarray,
        degrees: np.ndarray,
        starting_clique: np.ndarray,
        biggest_neighbourhood_exploration: int = 5,
        biggest_neighbourhood_intensification: int = 3,
        max_time: int = 60,
        verbose: bool = False,
        max_intensification_without_improve: int = 5,
        max_exploring_iterations_without_improve: int = 10,
        max_restarts_iterations_without_improve: int = 2,
        max_taboos: int = 10,
        **kwargs,
    ) -> None:
        """
        Arguments:
            - graph : the graph in which we are working
            - degrees : the degrees in our graph
            - starting_clique : the given starting solution
            - biggest_neighbourhood_exploration: the max i for V_i our neighbourhood
            - max_time : max_time to run the heuristic in seconds.
            - verbose : wether to print some informations during runtime or not.
            - max_intensification_without_improve: used for local descent on best.
            - max_exploring_iterations_without_improve: used for local descent on current.
        """
        self.graph = graph
        self.degrees = degrees
        self.starting_clique = starting_clique
        self.biggest_neighbourhood_exploration = biggest_neighbourhood_exploration
        self.biggest_neighbourhood_intensification = (
            biggest_neighbourhood_intensification
        )
        self.max_time = max_time
        self.verbose = verbose
        self.max_intensification_without_improve = max_intensification_without_improve
        self.max_exploring_iterations_without_improve = (
            max_exploring_iterations_without_improve
        )
        self.max_restarts_iterations_without_improve = (
            max_restarts_iterations_without_improve
        )
        self.max_taboos = max_taboos

    def reset_runtime_variables(self) -> None:
        self.start_time = time()
        self.best_clique = self.starting_clique
        self.current_clique = np.copy(self.best_clique)
        self.size_best_clique = np.sum(self.best_clique)
        self.time_best_found = 0
        self.iteration_best_clique = 0
        self.number_of_iterations = 0
        self.exploring_iterations_without_improve = 0
        self.restarts_iterations_without_improve = 0
        self.taboos_list = np.zeros(len(self.graph), dtype=bool)
        self.taboos_queue = Queue()
        self.taboos_queue_size = 0
        return

    def update_best_solution(self, new_clique: np.ndarray, new_size: int) -> None:
        self.time_best_found = time() - self.start_time
        self.size_best_clique = new_size
        self.best_clique = new_clique
        self.iteration_best_clique = self.number_of_iterations
        return

    def update_clique_only(self, new_clique: np.ndarray) -> None:
        self.best_clique = new_clique
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

    def update_taboos_with_nodes_to_delete(self, nodes_to_delete: list) -> None:
        for node in nodes_to_delete:
            self.taboos_queue.put(node)
            self.taboos_queue_size += 1
            if self.taboos_queue_size >= self.max_taboos:
                self.taboos_list[self.taboos_queue.get()] = False
            self.taboos_list[node] = True
        return

    def search_locally_from_best_clique(self) -> None:
        """
        Local descent from current best clique for at most max_intensification_without_improve.

        This is the intensification part.
        By default we stay between V_1 and V_3 of our best current solution.
        """
        current_biggest_neighbourhood_size = min(
            self.size_best_clique - 1, self.biggest_neighbourhood_intensification
        )
        for _ in range(self.max_intensification_without_improve):
            self.number_of_iterations += 1

            # Check several neighbourhoods
            for neighbourhood_size in range(1, current_biggest_neighbourhood_size + 1):
                # Remove some nodes from the clique taken randomly
                nodes_to_delete = i_th_nodes_removal_neighbour(
                    self.best_clique, i=neighbourhood_size
                )
                self.update_taboos_with_nodes_to_delete(nodes_to_delete=nodes_to_delete)
                new_clique = np.copy(self.best_clique)
                for node in nodes_to_delete:
                    new_clique[node] = 0
                # Search new best clique greedily from there.
                descending_random_from_clique_with_taboos(
                    clique=new_clique,
                    graph=self.graph,
                    degrees=self.degrees,
                    taboos_list=self.taboos_list,
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
        # Set biggest neighboourhood for current
        current_biggest_neighbourhood = min(
            np.sum(self.current_clique) - 1, self.biggest_neighbourhood_exploration
        )

        for neighbourhood_size in range(1, current_biggest_neighbourhood + 1):
            new_clique = np.copy(self.current_clique)
            # Remove some nodes from the clique taken randomly
            nodes_to_delete = i_th_nodes_removal_neighbour(
                self.current_clique, i=neighbourhood_size
            )
            self.update_taboos_with_nodes_to_delete(nodes_to_delete=nodes_to_delete)
            for node in nodes_to_delete:
                new_clique[node] = 0
            # Search new best clique greedily from there.
            descending_random_from_clique_with_taboos(
                clique=new_clique,
                graph=self.graph,
                degrees=self.degrees,
                taboos_list=self.taboos_list,
            )
            # See if the new result is better
            new_size = np.sum(new_clique)
            # In the exploration phase, being of same value than best is interesting
            # since we have already searched around the current best, so a new one
            # could yield better results.
            if new_size >= self.size_best_clique:
                if new_size == self.size_best_clique:
                    self.update_clique_only(new_clique=new_clique)
                # Else, if strictly better, update everything
                else:
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

        while (time() - self.start_time) < self.max_time:
            self.exploring_iterations_without_improve += 1

            # If exploration doesn't yield results, start again.
            if (
                self.exploring_iterations_without_improve
                > self.max_exploring_iterations_without_improve
            ):

                self.exploring_iterations_without_improve = 0
                self.restarts_iterations_without_improve += 1
                # If too many restart from best were made, restart from random.
                if (
                    self.restarts_iterations_without_improve
                    > self.max_restarts_iterations_without_improve
                ):
                    self.current_clique = np.zeros(len(self.graph))
                    # Set a random node to 1, then descend from it
                    self.current_clique[randint(len(self.graph) - 1)] = 1
                    descending_random_from_clique_with_taboos(
                        clique=self.current_clique,
                        graph=self.graph,
                        degrees=self.degrees,
                        taboos_list=self.taboos_list,
                    )
                # Otherwise restart from best
                else:
                    self.current_clique = np.copy(self.best_clique)
            else:
                pass

            # Exploring part
            self.explore_from_current_clique()

        return (
            self.best_clique,
            self.time_best_found,
            self.iteration_best_clique,
            self.number_of_iterations,
        )
