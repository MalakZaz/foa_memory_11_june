"""
===========================================================
 FILE: test_hybrid_memory.py
 PURPOSE: Test Bloom + Fitness Memory integration
===========================================================
"""

import numpy as np

from src.memory.bloom_filter import BloomFilter
from src.memory.fitness_memory import FitnessMemory

BAD_THRESHOLD = 0.90

if __name__ == "__main__":

    bloom = BloomFilter()

    memory = FitnessMemory()

    best_fit = 2000

    alpha_good = np.array([
        [1, 0],
        [0, 1]
    ])

    alpha_bad = np.array([
        [0, 1],
        [1, 0]
    ])

    print("\n========================================")
    print(" HYBRID MEMORY TEST")
    print("========================================")

    # =====================================================
    # Good solution
    # =====================================================
    bloom.add(alpha_good)
    memory.add(alpha_good, 1900)

    print("\nGOOD familiar solution")

    if bloom.contains(alpha_good):

        stored_fit = memory.get_fitness(alpha_good)

        if stored_fit < BAD_THRESHOLD * best_fit:

            print("Decision: SKIP")

        else:

            print("Decision: RE-EVALUATE")

    # =====================================================
    # Bad solution
    # =====================================================
    bloom.add(alpha_bad)
    memory.add(alpha_bad, 1200)

    print("\nBAD familiar solution")

    if bloom.contains(alpha_bad):

        stored_fit = memory.get_fitness(alpha_bad)

        if stored_fit < BAD_THRESHOLD * best_fit:

            print("Decision: SKIP")

        else:

            print("Decision: RE-EVALUATE")

    # =====================================================
    # Novel solution
    # =====================================================
    alpha_novel = np.array([
        [1, 1],
        [0, 0]
    ])

    print("\nNOVEL solution")

    if bloom.contains(alpha_novel):

        print("Decision: Familiar")

    else:

        print("Decision: EVALUATE")

    print("\n========================================")