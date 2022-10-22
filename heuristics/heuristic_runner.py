from typing import Callable, Tuple
from pathlib import Path

from time import time
import pandas as pd
import numpy as np

from utils import (
    read_single_problem_from_path_as_adjacency,
    check_validity_for_adjacency,
)


class HeuristicRunner:
    # Setup variables
    instances_dir: Path = None
    instances: list[str] = None
    methods: list[Tuple[Callable, str]] = None
    known_bests: dict = None
    # Runtime variables
    instance_reading_times = None
    clique_creating_times = None
    total_times = None
    clique_sizes = None
    display_time_details = None
    # Display variables
    display_dataframe = None

    def __init__(
        self,
        instances_dir: Path,
        instances: list[str],
        methods: list[Tuple[Callable, str]],
        known_bests: dict,
        display_time_details: bool = False,
    ) -> None:
        self.instances_dir = instances_dir
        self.instances = instances
        self.methods = methods
        self.known_bests = known_bests
        self.display_time_details = display_time_details

        # Initialize display dataframe
        columns = ["instance", "method", "found", "bound", "time"]
        if display_time_details:
            columns += ["load(s)", "clique(s)"]

        self.display_dataframe = pd.DataFrame({column: [] for column in columns})
        return

    def run_all_methods_on_instance(self, instance: str) -> None:
        for method, method_name in self.methods:
            # Loading
            start_time = time()
            _, _, graph, degrees = read_single_problem_from_path_as_adjacency(
                instance_path=self.instances_dir / instance
            )
            end_of_read_time = time()
            instance_reading_time = end_of_read_time - start_time

            # Clique building
            clique = method(graph=graph, degrees=degrees)
            clique_size = np.sum(clique)
            end_of_clique_time = time()
            clique_creating_time = end_of_clique_time - end_of_read_time

            assert check_validity_for_adjacency(graph, clique)
            total_time = time() - start_time

            # Adding to display
            new_row = [
                instance,
                method_name,
                clique_size,
                self.known_bests.get(instance, "//"),
                total_time,
            ]
            if self.display_time_details:
                new_row += [instance_reading_time, clique_creating_time]

            self.display_dataframe.loc[len(self.display_dataframe)] = new_row
        return

    def run_all_methods_on_all_instances(self) -> None:
        for instance in self.instances:
            self.run_all_methods_on_instance(instance=instance)
        return

    def display_results(self) -> None:
        print(self.display_dataframe)
        return
