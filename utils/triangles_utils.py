from typing import Tuple

import numpy as np


def node_iterator(graph: np.ndarray):
    n = len(graph)
    in_triangle = np.zeros(
        n,
    )

    for u in range(n):
        if not (in_triangle[u]):
            neighbours = []
            for v in range(n):
                if graph[u][v] and not (in_triangle[v]):
                    neighbours.append(v)
            deg_u = len(neighbours)
            for i in range(deg_u):
                for j in range(i):
                    if graph[neighbours[i]][neighbours[j]]:
                        in_triangle[neighbours[i]] = 1
                        in_triangle[neighbours[j]] = 1
                        in_triangle[u] = 1
    return in_triangle


def build_subgraph_from_triangles(
    graph: np.ndarray, degrees: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle = node_iterator(graph)
    sub_select = [i for i in range(len(graph)) if triangle[i]]

    return triangle, graph[np.ix_(sub_select, sub_select)], np.take(degrees, sub_select)


def build_clique_in_grah_from_subgraph(
    triangle: np.ndarray, subgraph_clique: np.ndarray
) -> np.ndarray:
    size_of_whole_graph = len(triangle)
    clique = np.zeros(size_of_whole_graph)
    pos = 0
    for node in range(size_of_whole_graph):
        if triangle[node]:
            if subgraph_clique[pos]:
                clique[node] = 1
            pos += 1
    return clique
