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
