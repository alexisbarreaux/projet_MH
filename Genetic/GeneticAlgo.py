from random import random
import numpy as np
import math as mt

from utils.triangles_utils import node_iterator
from utils.validity_checking_utils import check_validity_for_adjacency


def crossover(p1, p2, alpha):
    n = len(p1)
    child = np.zeros(n)
    for bit in range(n):
        if random() > alpha:
            child[bit] = int(p1[bit] & p2[bit])
        else:
            child[bit] = int(p1[bit] | p2[bit])
    return (sum(child), child)


def path_relinking(p1, p2):
    pass


def genetic_with_crossover(graph: np.ndarray, alpha: float):
    triangle = node_iterator(graph)
    sub_select = [i for i in range(len(graph)) if triangle[i]]
    sub_graph = graph[np._ix(sub_select, sub_select)]

    n = len(triangle)
    iterations = n
    mu = n
    couples = int(mu / 2)
    fertility = mt.sqrt(mu)

    # {Creating initial population ; Individual = (size, vector)}
    population = []

    for _ in range(iterations):
        # {Selection + Crossover}
        parents = np.random.shuffle(population)
        childs = []

        for p in range(couples):
            for _ in range(fertility):
                child = crossover(parents[p], parents[p + int(mu / 2)], alpha)
                if check_validity_for_adjacency(sub_graph, child[1]):
                    childs.append(child)

        # {Update}
        population = np.sort(np.concatenate((population, childs)))[-mu:]

    best = max(population)
    clique = []
    pos = 0
    for node in range(n):
        if triangle[node]:
            if best[pos]:
                clique.append(node)
            pos += 1
    return (clique, best[0])


def genetic_with_path_relinking(graph: np.ndarray, alpha: float):
    pass
