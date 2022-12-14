from typing import Tuple, TextIO

import numpy as np
from scipy.sparse import csr_array


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


def initialize_degrees(number_of_nodes: int) -> np.ndarray:
    """
    Initialize a 0 array for the degrees of the nodes.
    """
    return np.zeros(number_of_nodes)


def update_degrees_with_edge(
    first_edge_node: int, second_edge_node: int, degrees: np.ndarray
) -> None:
    degrees[first_edge_node] += 1
    degrees[second_edge_node] += 1
    return


def initialize_adjacency_matrix(number_of_nodes: int) -> np.ndarray:
    # Int 8 gives a smaller memory usage than the base use of int64
    return np.zeros((number_of_nodes, number_of_nodes), dtype=np.int8)


def update_adjacency_matrix_with_edge(
    first_edge_node: int, second_edge_node: int, adjacency_matrix: np.ndarray
) -> None:
    adjacency_matrix[first_edge_node][second_edge_node] = np.int8(1)
    adjacency_matrix[second_edge_node][first_edge_node] = np.int8(1)
    return


def read_single_problem_from_file_as_adjacency(
    instance_file: TextIO,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Used to read a file containing a unique problem.

    Returns:
        number of nodes
        number of edges
        graph as adjacency matrix
        degrees as an array
    """
    number_of_nodes, number_of_edges = read_instance_first_line(instance_file)
    adjacency_matrix = initialize_adjacency_matrix(number_of_nodes)
    degrees = initialize_degrees(number_of_nodes)

    for _ in range(number_of_edges):
        first_edge_node, second_edge_node = read_instance_edge_line(instance_file)
        update_adjacency_matrix_with_edge(
            first_edge_node, second_edge_node, adjacency_matrix
        )
        update_degrees_with_edge(first_edge_node, second_edge_node, degrees)

    return number_of_nodes, number_of_edges, adjacency_matrix, degrees


def read_single_problem_from_path_as_adjacency(
    instance_path: str,
) -> Tuple[int, int, np.ndarray]:
    """
    Used to read a file containing a unique problem from given path.
    """
    with open(instance_path, "r") as instance_file:
        return read_single_problem_from_file_as_adjacency(instance_file)


def initialize_sparse_array(
    number_of_nodes, number_of_edges
) -> Tuple[np.ndarray, list, list]:
    """
    Indptr will be of size n + 1 but we don't know the values so we only put zeroes
    at first.
    For indices and data they will be of size m, but for indices we don't
    know the value yet so initialize it to the empty list. For data however, since
    our edges are not valued, we know it will have only 1 as a value.
    """
    return (
        np.zeros(number_of_nodes + 1),
        [],
        np.ones(number_of_edges, dtype=np.int8),
    )


def update_sparse_with_edge(
    first_edge_node: int, second_edge_node: int, indptr: list, indices: list
) -> None:
    # First edge is noted i in comments and the second j, meaning the considered arc is (i,j)
    # We know there is an indice more for row i to be stored, so increment in indptr the end position
    # of row, meaning the value in position i + 1. If it was 0 it needs to become the value to the left
    # + 1, else it is just incremented
    if indptr[first_edge_node + 1] == 0:
        indptr[first_edge_node + 1] = indptr[first_edge_node] + 1
    else:
        indptr[first_edge_node + 1] += 1
    # We also need to put the right column position for the second edge in indices
    indices.append(second_edge_node)

    return


def read_single_problem_from_file_as_sparse(
    instance_file: TextIO,
) -> Tuple[int, int, csr_array, np.ndarray]:
    """
    Used to read a file containing a unique problem.

    Returns:
        number of nodes
        number of edges
        graph as sparse csr array
        degrees as an array
    """
    number_of_nodes, number_of_edges = read_instance_first_line(instance_file)
    indptr, indices, data = initialize_sparse_array(number_of_nodes, number_of_edges)
    degrees = initialize_degrees(number_of_nodes)

    for _ in range(number_of_edges):
        first_edge_node, second_edge_node = read_instance_edge_line(instance_file)
        update_sparse_with_edge(first_edge_node, second_edge_node, indptr, indices)
        update_degrees_with_edge(first_edge_node, second_edge_node, degrees)

    return (
        number_of_nodes,
        number_of_edges,
        csr_array(
            (data, np.array(indices), indptr), shape=(number_of_nodes, number_of_nodes)
        ),
        degrees,
    )


def read_single_problem_from_path_as_sparse(
    instance_path: str,
) -> Tuple[int, int, csr_array]:
    """
    Used to read a file containing a unique problem from given path and
    return the parameters, with the graph being represented as a scipy
    sparse matrix.
    """

    with open(instance_path, "r") as instance_file:
        return read_single_problem_from_file_as_sparse(instance_file)


def read_single_problem_from_path_as_sparse_from_adjacency(
    instance_path: str,
) -> Tuple[int, int, csr_array]:
    """
    Used to read a file containing a unique problem from given path and
    return the parameters, with the graph being represented as a scipy
    sparse matrix.
    """
    (
        number_of_nodes,
        number_of_edges,
        adjacency_matrix,
        degrees,
    ) = read_single_problem_from_path_as_adjacency(instance_path)
    return number_of_nodes, number_of_edges, csr_array(adjacency_matrix), degrees
