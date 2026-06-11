"""
===========================================================
 FILE: seed.py
 MODULE: Reproducibility Control
===========================================================
"""

import numpy as np
import random
import os


# ===========================================================
# GLOBAL SEED SETTER
# ===========================================================
def set_seed(seed: int = 42):
    """
    Ensure full reproducibility of FOA experiments.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)


# ===========================================================
# OPTIONAL: deterministic mode flag
# ===========================================================
def enable_deterministic_mode():
    """
    Force deterministic behavior (useful for paper experiments)
    """

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"