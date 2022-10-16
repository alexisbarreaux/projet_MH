from typing import Callable, Tuple
from pathlib import Path

from time import time
import pandas as pd


class HeuristicRunner:
    # Setup variables
    files_dir: Path = None
    files: list[str] = None
    methods: list[Tuple[Callable, str]] = None
    # Runtime variables
    instance_reading_times = None
    clique_creating_times = None
    total_times = None
    clique_sizes = None
    # Display variables
    display_dataframe = None

    def __init__(
        self,
        files_dir: Path,
        files: list[str],
        methods: list[Tuple[Callable, str]],
        **kwargs
    ) -> None:
        self.files_dir = files_dir
        self.files = files
        self.methods = methods

        for argument in kwargs:
            setattr(self, argument, kwargs[argument])

        # Initialize display dataframe
        columns = ["file", "method", "clique", "best", "total(s)"]
        for optional_attribute in ["load(s)", "clique(s)"]:
            if getattr(kwargs, optional_attribute, False):
                columns.append(optional_attribute)
            else:
                continue

        self.display_dataframe = pd.DataFrame({column: [] for column in columns})
        return

    def run_method_on_all_files(self, method: Callable, method_name: str) -> None:
        for file in self.files:
            method(file)
        return

    def run_all_methods_on_files(self) -> None:
        for (method, method_name) in self.methods:
            self.run_method_on_all_files(method=method, method_name=method_name)
        return


"""

for file in BASE_INSTANCES_FILES:
        # File loading
        start_time = time()
        _, _, graph, degrees = read_single_problem_from_path_as_adjacency(
            instance_path=BASE_INSTANCES_DIR / file
        )
        end_of_read_time = time()
        instance_reading_times.append(end_of_read_time - start_time)

        # Clique building
        clique = method(graph=graph, degrees=degrees)
        clique_sizes.append(np.sum(clique))
        end_of_clique_time = time()
        clique_creating_times.append(end_of_clique_time - end_of_read_time)

        assert check_validity_for_adjacency(graph, clique)
        total_times.append(time() - start_time)

    # At the end of each method, add visual separator
    for display_list in [
        files,
        instance_reading_times,
        clique_creating_times,
        total_times,
        clique_sizes,
        methods,
    ]:
        display_list.append("/////")

# Display for most basic heuristics

display_dataframe = pd.DataFrame(
    {
        "file": files,
        "method": methods,
        "clique size": clique_sizes,
        "instance time": instance_reading_times,
        "clique time": clique_creating_times,
        "total time": total_times,
    }
)
print(display_dataframe)
"""
