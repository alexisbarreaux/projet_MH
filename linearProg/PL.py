from docplex.mp.model import Model
import numpy as np


def LP(graph: np.ndarray):
    model = Model(name="naive_cplex")

    n = len(graph)

    x_list = [
        model.continuous_var(name="x_" + i, lb=0, ub=1) for i in range(n)
    ]  # Creating the variable x_i for each i in V

    for i in range(n):  # Adding the constraints of the problem
        for j in range(i):
            if not (graph[i][j]):
                model.add_constraint(x_list[i] + x_list[j] <= 1)

    model.maximize(sum(x_list))  # Define the objective function

    model.print_information()

    # model.set_time_limit(60)

    model.solve()

    model.print_solution()

    return int(model.objective_value)
