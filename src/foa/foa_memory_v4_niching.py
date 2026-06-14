"""
===========================================================
 FILE: foa_memory_v4_niching.py
 MODULE: Niching + Time-Sensitive Memory FOA V4
===========================================================

PURPOSE
-------
FOA-Memory V4 combines:
1. Multi-niche / multi-swarm exploration
2. Local time-sensitive Bloom memory per niche
3. Local fitness memory per niche
4. Global best solution selection

Each niche has its own memory, which preserves diversity and
prevents all flies from being guided by a single global memory.
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
    BLOOM_FAMILIARITY_THRESHOLD,
    NUM_NICHES
)


def binarize(X):
    """
    Convert continuous FOA position into binary allocation.
    """
    return (X > 0.5).astype(int)


def split_into_niches(X, num_niches):
    """
    Split the population into independent niches.
    """

    return np.array_split(X, num_niches)


def run_foa_memory_v4_niching(iterations,
                              pop_size,
                              N,
                              M,
                              eta,
                              scenario,
                              fitness_fn,
                              num_niches=NUM_NICHES,
                              bad_threshold=BAD_THRESHOLD,
                              familiarity_threshold=BLOOM_FAMILIARITY_THRESHOLD):
    """
    FOA-Memory V4 with niching and local memories.
    """

    X = initialize_population(pop_size, N, M)

    niches = split_into_niches(
        X,
        num_niches
    )

    bloom_memories = [
        TimeSensitiveBloomFilter()
        for _ in range(num_niches)
    ]

    fitness_memories = [
        FitnessMemory()
        for _ in range(num_niches)
    ]

    global_best_X = X[0].copy()
    global_best_alpha = binarize(global_best_X)
    global_best_fit = -np.inf

    best_history = []
    skipped_history = []
    bloom_occupancy_history = []
    novelty_history = []
    memory_size_history = []

    for iteration in range(iterations):

        total_skipped = 0
        iteration_novelties = []
        iteration_occupancies = []
        iteration_memory_sizes = []

        updated_niches = []

        for niche_id, niche in enumerate(niches):

            bloom = bloom_memories[niche_id]
            fitness_memory = fitness_memories[niche_id]

            bloom.recover()

            local_best_X = niche[0].copy()
            local_best_fit = -np.inf

            skipped = 0
            novelty_values = []

            for i in range(len(niche)):

                alpha = binarize(niche[i])

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

                    if global_best_fit > 0 and stored_fit < bad_threshold * global_best_fit:
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

                if fit > local_best_fit:
                    local_best_fit = fit
                    local_best_X = niche[i].copy()

                if fit > global_best_fit:
                    global_best_fit = fit
                    global_best_X = niche[i].copy()
                    global_best_alpha = alpha.copy()

            total_skipped += skipped

            if len(novelty_values) > 0:
                iteration_novelties.append(
                    float(np.mean(novelty_values))
                )

            iteration_occupancies.append(
                bloom.occupancy()
            )

            iteration_memory_sizes.append(
                fitness_memory.size()
            )

            updated_niche = update(
                niche,
                local_best_X,
                eta
            )

            updated_niches.append(updated_niche)

        niches = updated_niches

        best_history.append(global_best_fit)
        skipped_history.append(total_skipped)

        bloom_occupancy_history.append(
            float(np.mean(iteration_occupancies))
        )

        novelty_history.append(
            float(np.mean(iteration_novelties))
            if len(iteration_novelties) > 0
            else 0.0
        )

        memory_size_history.append(
            int(np.sum(iteration_memory_sizes))
        )

    return (
        global_best_alpha,
        global_best_fit,
        best_history,
        skipped_history,
        bloom_occupancy_history,
        memory_size_history,
        novelty_history
    )