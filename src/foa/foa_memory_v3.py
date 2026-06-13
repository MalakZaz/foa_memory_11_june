"""
===========================================================
 FILE: foa_memory_v3.py
 MODULE: Time-Sensitive Bloom + Fitness-Aware FOA-Memory V3
===========================================================

PURPOSE
-------
FOA-Memory V3 combines:

1. Time-Sensitive Bloom Filter:
   - detects whether a solution is novel or familiar;
   - allows gradual recovery of novelty over time.

2. Fitness Memory:
   - stores the fitness value of evaluated solutions.

Decision rule:
    - novel solution          -> evaluate
    - familiar good solution  -> re-evaluate
    - familiar bad solution   -> skip
===========================================================
"""

import numpy as np

from src.foa.population import initialize_population
from src.foa.movement import update
from src.core.repair import repair_solution

from src.memory.time_sensitive_bloom_filter import TimeSensitiveBloomFilter
from src.memory.fitness_memory import FitnessMemory

from src.config import (
    USE_QOS,
    BAD_THRESHOLD,
    BLOOM_FAMILIARITY_THRESHOLD
)


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
                      bad_threshold=BAD_THRESHOLD,
                      familiarity_threshold=BLOOM_FAMILIARITY_THRESHOLD):
    """
    Hybrid time-sensitive Bloom and fitness-aware FOA.
    """

    X = initialize_population(pop_size, N, M)

    bloom = TimeSensitiveBloomFilter()
    fitness_memory = FitnessMemory()

    best_X = X[0].copy()
    best_alpha = binarize(best_X)
    best_fit = -np.inf

    best_history = []
    skipped_history = []
    bloom_occupancy_history = []
    memory_size_history = []
    novelty_history = []

    for iteration in range(iterations):

        # Time effect: novelty progressively recovers.
        bloom.recover()

        skipped = 0
        novelty_values = []

        for i in range(pop_size):

            alpha = binarize(X[i])

            alpha = repair_solution(
                alpha,
                scenario["b"],
                scenario["B"],
                h=scenario["h"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                use_qos=USE_QOS
            )

            novelty = bloom.novelty_score(alpha)
            novelty_values.append(novelty)

            is_familiar = novelty < familiarity_threshold

            if is_familiar and fitness_memory.contains(alpha):

                stored_fit = fitness_memory.get_fitness(alpha)

                if best_fit > 0 and stored_fit < bad_threshold * best_fit:
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

            bloom.add(alpha)
            fitness_memory.add(alpha, fit)

            if fit > best_fit:
                best_fit = fit
                best_X = X[i].copy()
                best_alpha = alpha.copy()

        best_history.append(best_fit)
        skipped_history.append(skipped)
        bloom_occupancy_history.append(bloom.occupancy())
        memory_size_history.append(fitness_memory.size())

        if len(novelty_values) > 0:
            novelty_history.append(float(np.mean(novelty_values)))
        else:
            novelty_history.append(0.0)

        X = update(X, best_X, eta)

    return (
        best_alpha,
        best_fit,
        best_history,
        skipped_history,
        bloom_occupancy_history,
        memory_size_history,
        novelty_history
    )