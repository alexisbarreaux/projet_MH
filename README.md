# projet_MH

## Structure of the repository
This repository contains the following folders and main subfolders:
- `genetics` : work done on implementing genetic metaheuristics
- `heuristics` : the developped heuristics for this projet and a runner class to easily run several methods on several files at once.
- `instances` : the files containing the instances/graphs on which we tested our code, it contains two subfolders
    - `project_instances`: the instances we were given as part of the subject
    - `other_instances`: other (bigger) instances we tested on
- `linearProg` : code regarding implementation of the PL and PLNE linked to the maximum clique problem, to be able to have bounds on some instances.
- `meta_heuristics` : the non genetic meta heuristics implemented
- `notebooks` : several notebooks to keep track of the tests we made and how we created the corresponding results files. It contains a subfolder:
    - `results`: for all the non main results we computed along the way.

## Running on an instance
In order for one to run our heuristics or meta heuristics on an instance, from the root of this project one must:

- create a venv
    
```python
python -m venv .
```
- install the required packages
```python
pip install -r requirements.txt
```
- go to or create a notebook and use the need code, the most importing of it:
```python
# imports
from pathlib import Path

import numpy as np

from utils import (
    read_single_problem_from_path_as_adjacency,
    check_validity_for_adjacency
)
from constants import (
    BASE_INSTANCES_FILES,
    OTHER_INSTANCES_FILES,
    OTHER_INSTANCES_BEST_KNOWN,
    BASE_INSTANCES_BEST_KNOWN,
    ALL_BEST_KNOWN,
)
from heuristics import descending_degree_glutonous_heuristic
from meta_heuristics import ExploringMetaHeuristicRunner, ExploringTaboosMetaHeuristicRunner, ExploringTaboosRestartMetaHeuristicRunner

# Constants
ROOT_DIR = Path.cwd().parent
# Instances pathes
INSTANCES_DIR = ROOT_DIR / "instances"
BASE_INSTANCES_DIR = INSTANCES_DIR / "project_instances"
OTHER_INSTANCES_DIR = INSTANCES_DIR / "other_instances"

# Loading an instance and getting its parameters
number_of_nodes, number_of_edges, graph, degrees = read_single_problem_from_path_as_adjacency(
        instance_path=BASE_INSTANCES_DIR / instance
    )
# Running the determinist heuristic on it
starting_clique = descending_degree_glutonous_heuristic(
        graph=graph, degrees=degrees
    )
# Setting meta parameters
max_time = 120
max_intensification_without_improve = 5
max_exploring_iterations_without_improve = 3 * max_intensification_without_improve
max_restarts_iterations_without_improve = max_intensification_without_improve
# Creating runner with parameters for one of our meta heuristics
runner = ExploringTaboosMetaHeuristicRunner(
            graph=graph,
            degrees=degrees,
            starting_clique=starting_clique,
            max_time=max_time,
            max_intensification_without_improve=max_intensification_without_improve,
            max_exploring_iterations_without_improve=max_exploring_iterations_without_improve,
            max_restarts_iterations_without_improve= max_restarts_iterations_without_improve
        )
# Running it
(
clique,
time_best_found,
iteration_of_best,
number_of_iterations,
) = runner.meta_heuristic()
# Checking that the result is a valid clique
assert check_validity_for_adjacency(graph, clique)
```