"""
===========================================================
 FILE: population.py
 MODULE: Initial Population Generation
 PROJECT: FOA-CRN V0
===========================================================

DESCRIPTION
-----------
Generates the initial population of candidate spectrum
allocations for the Fruit Fly Optimization Algorithm (FOA).

Each individual is represented by an allocation matrix:

    alpha[i,p] = 1
        SU i assigned to channel p

    alpha[i,p] = 0
        otherwise

The exclusivity constraint is enforced during generation:

    Σ_p alpha[i,p] ≤ 1

===========================================================
"""

import numpy as np
from config import N, M, POPULATION

# ===========================================================
# GENERATE ONE INDIVIDUAL
# ===========================================================
def generate_individual(N, M):
    """
    Generate one random allocation matrix.

    Parameters
    ----------
    N : int
        Number of secondary users

    M : int
        Number of PU channels

    Returns
    -------
    alpha : ndarray (N,M)
    """

    alpha = np.zeros((N, M), dtype=int)

    for i in range(N):

        # 0 means "not served"
        assigned_channel = np.random.randint(0, M + 1)

        if assigned_channel > 0:
            alpha[i, assigned_channel - 1] = 1

    return alpha


# ===========================================================
# INITIAL POPULATION
# ===========================================================
def initialize_population(pop_size, N, M):

    population = np.array(
        [generate_individual(N, M)
         for _ in range(pop_size)]
    )

    return population