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
OTHER_INSTANCES_FILES = [
    "brock800_4.txt",
    "C2000.5.txt",
    "C500.9.txt",
    "gen400_p0.9_75.txt",
    "p_hat1500-3.txt",
    "p_hat700-1txt.txt",
]
