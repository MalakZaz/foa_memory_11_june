
"""
F:\Amel TH\implémentation 08 mai 2026\foa_crn_v0\foa
movement.py - Déplacement des mouches
"""

import numpy as np
from config import N, M, ETA


def move_towards_best(position, best_position):
    """
    Met à jour la position selon la règle FOA.
    X_new = (X_old * r1) + (X_best * (1 - r1)) + (ETA * r2)
    """
    r1 = np.random.rand(N, M)
    r2 = np.random.rand(N, M)
    
    new_position = (position * r1) + (best_position * (1 - r1)) + (ETA * r2)
    return np.clip(new_position, 0, 1)


def compute_distance(pos1, pos2):
    """Calcule la distance Euclidienne entre deux positions."""
    return np.sqrt(np.sum((pos1 - pos2) ** 2))