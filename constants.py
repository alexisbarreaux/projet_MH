from pathlib import Path

# Directory constants
ROOT_DIR = Path.cwd()
# Instances pathes
INSTANCES_DIR = ROOT_DIR / "instances"
BASE_INSTANCES_DIR = INSTANCES_DIR / "project_instances"
OTHER_INSTANCES_DR = INSTANCES_DIR / "other_instances"
# Instances files groups
BASE_INSTANCES_FILES = [
    "brock200_2.col",
    "dsjc125.1.col",
    "random-10.col",
    "random-100.col",
    "random-40.col",
    "random-70.col",
]
