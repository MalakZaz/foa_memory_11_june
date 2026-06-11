"""
===========================================================
 FILE: metrics.py
 MODULE: FOA-CRN Performance Metrics
===========================================================
"""

import numpy as np


# ===========================================================
# FEASIBILITY RATE
# ===========================================================
def feasibility_rate(feasible_list):
    """
    Ratio of feasible solutions in population.
    """
    return np.mean(feasible_list)


# ===========================================================
# AVERAGE FITNESS
# ===========================================================
def average_fitness(fitness_list):
    """
    Mean population fitness.
    """
    return np.mean(fitness_list)


# ===========================================================
# BEST IMPROVEMENT GAIN
# ===========================================================
def improvement_gain(best_history):
    """
    Measures convergence speed.

    gain = (f_best_final - f_best_initial) / |f_best_initial|
    """
    if len(best_history) < 2:
        return 0.0

    return (best_history[-1] - best_history[0]) / (abs(best_history[0]) + 1e-9)


# ===========================================================
# DIVERSITY METRIC (OPTIONAL BUT IMPORTANT SCIENTIFICALLY)
# ===========================================================
def population_diversity(X):
    """
    Measures exploration capability of FOA.

    Uses mean pairwise Hamming distance.
    """
    pop_size = len(X)
    dist = 0.0
    count = 0

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist += np.sum(X[i] != X[j])
            count += 1

    return dist / (count + 1e-9)