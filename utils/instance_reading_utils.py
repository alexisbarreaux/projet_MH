from typing import Tuple, List, TextIO

import numpy as np


def read_instance_first_line(instance_file: TextIO) -> Tuple[int, int]:
    """
    Discard unneeded 'p' and 'edge' at start.
    """
    first_line_splitted = instance_file.readline().split(" ")
    return int(first_line_splitted[2]), int(first_line_splitted[3])


def read_instance_edge_line(instance_file: TextIO) -> Tuple[int, int]:
    """
    Discard unneeded 'e' at start.
    """
    line_splitted = instance_file.readline().split(" ")
    # We need the minus ones since the source files are written with
    # node from 1 to n.
    return int(line_splitted[1]) - 1, int(line_splitted[2]) - 1


def initialize_adjacency_matrix(number_of_nodes: int) -> np.ndarray:
    # Int 8 gives a smaller memory usage than the base use of int64
    return np.zeros((number_of_nodes, number_of_nodes), dtype=np.int8)


def update_adjacency_matrix_with_edge(
    first_edge_node: int, second_edge_node: int, adjacency_matrix: np.ndarray
) -> None:
    adjacency_matrix[first_edge_node][second_edge_node] = np.int8(1)
    return


def read_single_problem_from_file(
    instance_file: TextIO,
) -> Tuple[int, int, np.ndarray]:
    """
    Used to read a file containing a unique problem.

    There is a question of how to store the graph. First let us use a simple
    node-node adjacency matrix.
    """
    number_of_nodes, number_of_edges = read_instance_first_line(instance_file)
    adjacency_matrix = initialize_adjacency_matrix(number_of_nodes)

    for _ in range(number_of_edges):
        first_edge_node, second_edge_node = read_instance_edge_line(instance_file)
        update_adjacency_matrix_with_edge(
            first_edge_node, second_edge_node, adjacency_matrix
        )

    return number_of_nodes, number_of_edges, adjacency_matrix


def read_single_problem_from_path(
    instance_path: str,
) -> Tuple[str, int, int, int, List[int]]:
    """
    Used to read a file containing a unique problem from given path.
    """
    with open(instance_path, "r") as instance_file:
        return read_single_problem_from_file(instance_file)


if __name__ == "__main__":
    from constants import BASE_INSTANCES_DIR

    print(
        read_single_problem_from_path(
            instance_path=BASE_INSTANCES_DIR / "random-10.col"
        )
    )
