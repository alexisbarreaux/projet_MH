from math import ceil, log


def max_best_iterations_without_improve_parameter(clique_size: int) -> int:
    """
    max_best_iterations_without_improve is taken so that the probability
    of having not removed a node from the clique for V_1 of the first clique
    found with the heuristic is less than 1/2
    """
    return ceil(log(0.5) / log(1 - 1 / clique_size))
