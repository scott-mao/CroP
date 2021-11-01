import os

import numpy as np

from utils.data_manager import DataManager



PROJ_NAME = "CroP"
WORKING_DIR_PATH = "."

# output
RESULTS_DIR = "results"
DATA_DIR = "data"
# GITIGNORED_DIR = "/nfs/students/rachwan/gitignored/gitignored"
GITIGNORED_DIR = "gitignored"

IMAGENETTE_DIR = os.path.join(".", "gitignored", "data", "imagenette-320")
IMAGEWOOF_DIR = os.path.join(".", "gitignored", "data", "imagewoof-320")
TINY_IMAGNET_DIR = os.path.join(".", "gitignored", "data", "tiny_imagenet")

CODEBASE_DIR = "codebase"
SUMMARY_DIR = "summary"
OUTPUT_DIR = "output"
MODELS_DIR = "models"
PROGRESS_DIR = "progress"
OUTPUT_DIRS = [OUTPUT_DIR, SUMMARY_DIR, CODEBASE_DIR, MODELS_DIR, PROGRESS_DIR]

DATA_MANAGER = DataManager(os.path.join(WORKING_DIR_PATH, GITIGNORED_DIR))
DATASET_PATH = os.path.join(DATA_MANAGER.directory, DATA_DIR)
RESULTS_PATH = os.path.join(DATA_MANAGER.directory, RESULTS_DIR)

# printing
PRINTCOLOR_PURPLE = '\033[95m'
PRINTCOLOR_CYAN = '\033[96m'
PRINTCOLOR_DARKCYAN = '\033[36m'
PRINTCOLOR_BLUE = '\033[94m'
PRINTCOLOR_GREEN = '\033[92m'
PRINTCOLOR_YELLOW = '\033[93m'
PRINTCOLOR_RED = '\033[91m'
PRINTCOLOR_BOLD = '\033[1m'
PRINTCOLOR_UNDERLINE = '\033[4m'
PRINTCOLOR_END = '\033[0m'

MODELS_DIR = "models"
LOSS_DIR = "losses"
CRITERION_DIR = "criterions"
NETWORKS_DIR = "networks"
TRAINERS_DIR = "trainers"
OPTIMS = "optim"
DATASETS = "datasets"
types = [LOSS_DIR, NETWORKS_DIR, CRITERION_DIR, TRAINERS_DIR]

TEST_SET = "test"
VALIDATION_SET = "validation"
TRAIN_SET = "train"

ZERO_SIGMA = -1 * 1e6

SNIP_BATCH_ITERATIONS = 5

HOYER_THERSHOLD = 1e-3

SMALL_POOL = (2, 2)
PROD_SMALL_POOL = np.prod(SMALL_POOL)
MIDDLE_POOL = (3, 3)
PROD_MIDDLE_POOL = np.prod(MIDDLE_POOL)
BIG_POOL = (5, 5)
PROD_BIG_POOL = np.prod(BIG_POOL)
NUM_WORKERS = 6
FLIP_CHANCE = 0.2
STRUCTURED_SINGLE_SHOT = [
    "SNAP",
    "SNAP_Res",
    "StructuredRandom",
    "CROPStructured",
    "CROPitStructured",
    "EarlyCROPitStructured"]
SINGLE_SHOT = [
    "SNIP",
    "GRASP",
    "CROP",
    "CROPit",
    "UnstructuredRandom",
    "EarlyCROP"
]
SINGLE_SHOT += STRUCTURED_SINGLE_SHOT
DURING_TRAINING = [
    "CROPStructuredDuring",
    "CROPStructuredDuringRes",
    "GateDecorators",
    "EfficientConvNets"
]

TIMEOUT = int(60 * 60 * 1.7)  # one hour and a 45 minutes max
STACK_NAME = "command_stack"
