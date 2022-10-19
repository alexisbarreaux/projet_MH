# The idea is to build a clique by using the nodes with the biggest degrees first.
# This is fully deterministic.
# Cliques are built as the list of nodes first, before being transformed to a 0-1 array
from typing import Tuple

import numpy as np

from utils import (
    order_nodes_in_descending_order,
    order_subset_of_nodes_in_descending_order,
    check_if_edge_exists_in_adjacency,
    delete_node_from_graph,
    get_degrees_in_adjacency,
    transform_node_clique_to_zero_one,
    check_clique_validity,
)


def descending_degree_glutonous_heuristic(
    graph: np.ndarray, degrees: np.ndarray
) -> list:
    """
    Start by reordering the nodes by degrees, then starting from the one with the
    biggest degree, try to build the biggest possible clique.

    O(n**2) because we need to check for each node except the first if the possibly all
    previous nodes are neighbours, each of which is done in O(1).
    """
    ordered_nodes = order_nodes_in_descending_order(degrees=degrees)

    # Start clique with node of biggest degree
    clique: list = [ordered_nodes[0]]
    clique_size: int = 1
    # Possible candidates are all other nodes
    candidates = ordered_nodes[1:]
    for candidate_node in candidates:
        # If at some point the remaining degrees are less than the number
        # of elements in the clique, no more node can be added, and since they
        # are ordered once one node has a degree which is too small, all the
        # next ones too
        if degrees[candidate_node] < clique_size:
            return transform_node_clique_to_zero_one(len(graph), clique)
        elif np.sum(np.take(graph[candidate_node], indices=clique)) == clique_size:
            clique.append(candidate_node)
            clique_size += 1
        else:
            continue

    return transform_node_clique_to_zero_one(len(graph), clique)


def descending_degree_glutonous_heuristic_row_sum_version(
    graph: np.ndarray, degrees: np.ndarray
) -> list:
    """
    Start by reordering the nodes by degrees, then starting from the one with the
    biggest degree, try to build the biggest possible clique.

    O(n**2) because we need to check for each node except the first if the possibly all
    previous nodes are neighbours, each of which is done in O(1).
    """
    ordered_nodes = order_nodes_in_descending_order(degrees=degrees)

    # Start clique with node of biggest degree
    clique: np.ndarray = np.zeros(len(graph))
    clique[ordered_nodes[0]] = 1
    clique_size: int = 1
    # Possible candidates are all other nodes
    for candidate_node in ordered_nodes[1:]:
        # If at some point the remaining degrees are less than the number
        # of elements in the clique, no more node can be added, and since they
        # are ordered once one node has a degree which is too small, all the
        # next ones too
        if degrees[candidate_node] < clique_size:
            return clique
        elif check_clique_validity(
            graph=graph,
            degrees=degrees,
            clique=clique,
            clique_size=clique_size,
            new_node=candidate_node,
        ):
            clique[candidate_node] = 1
            clique_size += 1
        else:
            continue

    return clique


def descending_degree_glutonous_heuristic_from_clique(
    clique: np.ndarray, graph: np.ndarray, degrees: np.ndarray
) -> None:
    """
    From a base clique attempt to build a bigger one by regarding the nodes not
    yet in the clique by descending order.
    Updates the base clique in place.
    """
    clique_size = np.sum(clique)
    clique_nodes = [i for i in range(len(clique)) if clique[i] == 1]
    # Take nodes not yet in the clique and order them by descending degree
    candidates = [i for i in range(len(clique)) if clique[i] == 0]
    ordered_candidates = order_subset_of_nodes_in_descending_order(
        nodes=candidates, degrees=degrees
    )

    for candidate_node in ordered_candidates:
        # If at some point the remaining degrees are less than the number
        # of elements in the clique, no more node can be added, and since they
        # are ordered once one node has a degree which is too small, all the
        # next ones too
        if degrees[candidate_node] < clique_size:
            return
        elif np.all(
            [
                check_if_edge_exists_in_adjacency(
                    graph=graph, first_node=candidate_node, second_node=clique_node
                )
                for clique_node in clique_nodes
            ]
        ):
            clique_nodes.append(candidate_node)
            clique[candidate_node] = 1
            clique_size += 1
        else:
            continue

    return


def update_residual_graph_and_reorder_nodes(
    residual_graph: np.ndarray, removed_node: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove a node (all his edges) from a graph, recompute degrees and then order
    nodes again by descending order.
    """
    # Remove edges to/from last added node in the clique
    delete_node_from_graph(residual_graph, node=removed_node)
    # Recompute degrees
    degrees = get_degrees_in_adjacency(residual_graph)
    # Order them
    ordered_nodes = order_nodes_in_descending_order(degrees=degrees)
    return degrees, ordered_nodes


def dynamic_descending_degree_glutonous_heuristic(
    graph: np.ndarray, degrees: np.ndarray
) -> list:
    """
    Start by reordering the nodes by degrees, then starting from the one with the
    biggest degree, try to build the biggest possible clique. Each time a node is added,
    delete all his edges and reorder remaining nodes.

    O(n**2) because we need to check for each node except the first if the possibly all
    previous nodes are neighbours, each of which is done in O(1). The deleting of edges
    and recomputing of degrees is also O(n**2), reordering nodes is O(n*ln(n))
    """
    ordered_nodes = order_nodes_in_descending_order(degrees=degrees)
    residual_graph = np.copy(graph)
    clique = []

    # Store wether nodes were already tested or not.
    treated_nodes = np.zeros(len(graph), dtype=bool)

    while not all(treated_nodes):
        # Add first valid candidate
        for candidate_node in ordered_nodes:
            # If the node was already treated, go to next one
            if treated_nodes[candidate_node]:
                continue
            else:
                treated_nodes[candidate_node] = True
                if np.all(
                    [
                        check_if_edge_exists_in_adjacency(
                            graph=graph,
                            first_node=candidate_node,
                            second_node=clique_node,
                        )
                        for clique_node in clique
                    ]
                ):
                    clique.append(candidate_node)
                    degrees, ordered_nodes = update_residual_graph_and_reorder_nodes(
                        residual_graph=residual_graph, removed_node=candidate_node
                    )
                    # Break to restart loop on valid candidates order
                    break

    return transform_node_clique_to_zero_one(len(graph), clique)
