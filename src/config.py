import random
from pathlib import Path

import numpy as np


class Settings:
    SEED = 42
    VALIDATION_SEED = 20261121
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    OUTPUT_DIR = ROOT / "output"
    ID = "id"
    TARGET = "is_canceled"
    TEST_SIZE = 0.2
    N_FOLDS = 5


def seed_everything(seed=Settings.SEED):
    random.seed(seed)
    np.random.seed(seed)
