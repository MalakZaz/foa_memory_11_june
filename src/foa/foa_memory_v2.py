"""
===========================================================
 FILE: foa_memory_v2.py
 MODULE: Fitness-Aware FOA-Memory V2
===========================================================

PURPOSE
-------
FOA-Memory V2 avoids only familiar bad solutions.

A candidate is skipped if:
    - it has already been evaluated;
    - its stored fitness is lower than bad_threshold * best_fit.
===========================================================
"""

import numpy as np

from src.foa.population import initialize_population
from src.foa.movement import update
from src.memory.fitness_memory import FitnessMemory
from src.core.repair import repair_solution
from src.config import USE_QOS, BAD_THRESHOLD


def binarize(X):
    """
    Convert continuous FOA position into binary allocation.
    """
    return (X > 0.5).astype(int)


def run_foa_memory_v2(iterations,
                      pop_size,
                      N,
                      M,
                      eta,
                      scenario,
                      fitness_fn,
                      bad_threshold=BAD_THRESHOLD):
    """
    Fitness-aware memory-enhanced FOA.
    """

    X = initialize_population(pop_size, N, M)

    memory = FitnessMemory()

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

            # MKP + optional QoS repair
            alpha = repair_solution(
                alpha,
                scenario["b"],
                scenario["B"],
                h=scenario["h"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                use_qos=USE_QOS
            )

            if memory.contains(alpha):

                stored_fitness = memory.get_fitness(alpha)

                if best_fit > 0 and stored_fitness < bad_threshold * best_fit:
                    skipped += 1
                    continue

            fit = fitness_fn(
                alpha,
                h=scenario["h"],
                B=scenario["B"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                w=scenario["w"],
                d=scenario["d"]
            )

            memory.add(alpha, fit)

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