"""
population.py - Initialisation de la population
"""

import numpy as np
from config import N, M, POPULATION


def initialize_population():
    """Génère une population aléatoire de positions."""
    population = []
    for _ in range(POPULATION):
        position = np.random.rand(N, M)
        population.append(position)
    return population