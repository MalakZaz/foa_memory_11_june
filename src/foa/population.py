"""
===========================================================
 FILE: population.py
 MODULE: Initial Population Generation
===========================================================
"""

import numpy as np


def initialize_population(pop_size, N, M):
    """
    Generate a random continuous FOA population.

    Returns
    -------
    X : ndarray
        Population of shape (pop_size, N, M).
    """

    return np.random.rand(pop_size, N, M)