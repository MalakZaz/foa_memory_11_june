"""
===========================================================
 FILE: foa_memory_v3.py
 MODULE: Hybrid Bloom + Fitness-Aware FOA-Memory V3
===========================================================

PURPOSE
-------
FOA-Memory V3 combines:

1. Bloom Filter:
   - detects whether a solution is familiar.

2. Fitness Memory:
   - stores the fitness value of evaluated solutions.

Decision rule:
    - novel solution        -> evaluate
    - familiar good solution -> re-evaluate
    - familiar bad solution  -> skip

This version preserves the biological novelty-detection
motivation while adding fitness-aware avoidance.
===========================================================
"""

import numpy as np

from src.foa.population import initialize_population
from src.foa.movement import update

from src.core.repair import repair_solution

from src.memory.bloom_filter import BloomFilter
from src.memory.fitness_memory import FitnessMemory

from src.config import USE_QOS, BAD_THRESHOLD


def binarize(X):
    """
    Convert continuous FOA position into binary allocation.
    """
    return (X > 0.5).astype(int)


def run_foa_memory_v3(iterations,
                      pop_size,
                      N,
                      M,
                      eta,
                      scenario,
                      fitness_fn,
                      bad_threshold=BAD_THRESHOLD):
    """
    Hybrid Bloom + Fitness Memory FOA.

    Returns
    -------
    best_alpha : ndarray
        Best allocation matrix.

    best_fit : float
        Best fitness value.

    best_history : list
        Best fitness value at each iteration.

    skipped_history : list
        Number of skipped evaluations per iteration.

    bloom_occupancy_history : list
        Bloom filter occupancy per iteration.

    memory_size_history : list
        Number of exact fitness records stored.
    """

    X = initialize_population(pop_size, N, M)

    bloom = BloomFilter()
    fitness_memory = FitnessMemory()

    best_X = X[0].copy()
    best_alpha = binarize(best_X)
    best_fit = -np.inf

    best_history = []
    skipped_history = []
    bloom_occupancy_history = []
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

            # ------------------------------------------------
            # Hybrid memory decision
            # ------------------------------------------------
            is_familiar = bloom.contains(alpha)

            if is_familiar and fitness_memory.contains(alpha):

                stored_fit = fitness_memory.get_fitness(alpha)

                if best_fit > 0 and stored_fit < bad_threshold * best_fit:
                    skipped += 1
                    continue

            # ------------------------------------------------
            # Evaluate candidate
            # ------------------------------------------------
            fit = fitness_fn(
                alpha,
                h=scenario["h"],
                B=scenario["B"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                w=scenario["w"],
                d=scenario["d"]
            )

            # ------------------------------------------------
            # Update memories
            # ------------------------------------------------
            bloom.add(alpha)
            fitness_memory.add(alpha, fit)

            # ------------------------------------------------
            # Update best solution
            # ------------------------------------------------
            if fit > best_fit:
                best_fit = fit
                best_X = X[i].copy()
                best_alpha = alpha.copy()

        best_history.append(best_fit)
        skipped_history.append(skipped)
        bloom_occupancy_history.append(bloom.occupancy())
        memory_size_history.append(fitness_memory.size())

        X = update(X, best_X, eta)

    return (
        best_alpha,
        best_fit,
        best_history,
        skipped_history,
        bloom_occupancy_history,
        memory_size_history
    )