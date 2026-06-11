"""
===========================================================
 FILE: foa_memory_v1.py
 MODULE: Memory-Enhanced Fruit Fly Optimization Algorithm
===========================================================

PURPOSE
-------
FOA-Memory V1 extends FOA V0 by avoiding redundant
candidate evaluations using an exact solution memory.

This version:
    - stores already evaluated allocation matrices
    - skips duplicate candidates
    - tracks the number of skipped evaluations
===========================================================
"""

import numpy as np

from src.foa.population import initialize_population
from src.foa.movement import update
from src.memory.solution_memory import SolutionMemory


def binarize(X):
    """
    Convert continuous FOA position into binary allocation.
    """
    return (X > 0.5).astype(int)


def run_foa_memory_v1(iterations,
                      pop_size,
                      N,
                      M,
                      eta,
                      scenario,
                      fitness_fn):
    """
    Memory-enhanced FOA optimizer.

    Returns
    -------
    best_alpha : ndarray
    best_fit : float
    best_history : list
    skipped_history : list
    memory_size_history : list
    """

    X = initialize_population(pop_size, N, M)

    memory = SolutionMemory()

    best_X = X[0].copy()
    best_alpha = binarize(best_X)
    best_fit = -np.inf

    best_history = []
    skipped_history = []
    memory_size_history = []

    for iteration in range(iterations):

        skipped = 0

        for i in range(pop_size):

            alpha = binarize(X[i])

            if memory.contains(alpha):
                skipped += 1
                continue

            memory.add(alpha)

            fit = fitness_fn(
                alpha,
                h=scenario["h"],
                B=scenario["B"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                w=scenario["w"],
                d=scenario["d"]
            )

            if fit > best_fit:
                best_fit = fit
                best_X = X[i].copy()
                best_alpha = alpha.copy()

        best_history.append(best_fit)
        skipped_history.append(skipped)
        memory_size_history.append(memory.size())

        X = update(X, best_X, eta)

    return (
        best_alpha,
        best_fit,
        best_history,
        skipped_history,
        memory_size_history
    )