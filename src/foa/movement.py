"""
===========================================================
 FILE: movement.py
 MODULE: Fruit Fly Movement Operator
===========================================================
"""

import numpy as np


def move_towards_best(position,
                      best_position,
                      eta):
    """
    Update one fruit fly position.

    Parameters
    ----------
    position : ndarray
        Current fly position.

    best_position : ndarray
        Current best solution.

    eta : float
        Exploration intensity.

    Returns
    -------
    ndarray
        Updated position.
    """

    N, M = position.shape

    r1 = np.random.rand(N, M)
    r2 = np.random.rand(N, M)

    new_position = (
        position * r1
        + best_position * (1 - r1)
        + eta * r2
    )

    return np.clip(new_position, 0, 1)


def update(X,
           best_X,
           eta):
    """
    Update the whole population.
    """

    new_X = np.zeros_like(X)

    for i in range(len(X)):

        new_X[i] = move_towards_best(
            X[i],
            best_X,
            eta
        )

    return new_X


def compute_distance(pos1,
                     pos2):
    """
    Euclidean distance between two flies.
    """

    return np.linalg.norm(
        pos1 - pos2
    )