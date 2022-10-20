import random as rd
import numpy as np
import math as mt
from operator import itemgetter
import statistics

from utils.triangles_utils import node_iterator
from utils.validity_checking_utils import check_validity_for_adjacency
from heuristics.descending_random import descending_random_from_clique


def crossover(p1: np.ndarray, p2: np.ndarray, alpha: float):
    n = len(p1[1])
    child = np.zeros(n)
    for i in range(n):
        if rd.random() > alpha:
            child[i] = int(p1[1][i] & p2[1][i])
        else:
            child[i] = int(p1[1][i] | p2[1][i])
    return (sum(child), child)


def genetic_with_crossover(graph: np.ndarray, degrees: np.ndarray, alpha: float):
    triangle = node_iterator(graph)
    sub_select = [i for i in range(len(graph)) if triangle[i]]
    sub_graph = graph[np.ix_(sub_select, sub_select)]

    n = len(triangle)
    sub_n = len(sub_select)

    iterations = sub_n
    mu = sub_n
    couples = int(mu / 2)
    fertility = int(mt.sqrt(mu))

    # {Creating initial population ; Individual = (size, vector)}
    sub_degrees = np.take(degrees, indices=sub_select)
    population = []
    for _ in range(mu):
        node = rd.randint(0, sub_n - 1)
        initial_clique = np.zeros(sub_n)
        initial_clique[node] = 1
        descending_random_from_clique(initial_clique, sub_graph, sub_degrees)
        initial_clique = np.array(list(map(int, initial_clique)))
        population.append((sum(initial_clique), initial_clique))

    for _ in range(iterations):
        # {Random selection + Crossover}
        parents = np.copy(population)
        np.random.shuffle(parents)
        children = []

        for p in range(couples):
            for _ in range(fertility):
                child = crossover(parents[p], parents[p + int(mu / 2)], alpha)
                if check_validity_for_adjacency(sub_graph, child[1]):
                    children.append(child)

        # {Update}
        if children:
            print("yay")
            population = sorted(
                np.concatenate((population, children)), key=itemgetter(0)
            )[-mu:]
        # {TESTS}
        print(statistics.mean(list(zip(*population))[0]))
        # print(max(population, key=itemgetter(0))[0])

    best = max(population, key=itemgetter(0))
    clique = np.zeros(n)
    pos = 0
    for node in range(n):
        if triangle[node]:
            if best[1][pos]:
                clique[node] = 1
            pos += 1
    return clique
