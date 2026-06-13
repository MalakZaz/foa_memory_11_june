"""
===========================================================
 FILE: test_fitness_memory.py
 PURPOSE: Test exact fitness memory
===========================================================
"""

import numpy as np

from src.memory.fitness_memory import FitnessMemory
from src.config import N, M


if __name__ == "__main__":

    memory = FitnessMemory()

    alpha_1 = np.zeros((N, M), dtype=int)
    alpha_1[0, 0] = 1
    alpha_1[1, 1] = 1

    alpha_2 = np.zeros((N, M), dtype=int)
    alpha_2[2, 2] = 1

    fitness_1 = 1500.0

    print("\n========================================")
    print(" FITNESS MEMORY TEST")
    print("========================================")

    print("Before adding alpha_1:")
    print("Contains alpha_1:", memory.contains(alpha_1))
    print("Fitness alpha_1 :", memory.get_fitness(alpha_1))

    memory.add(alpha_1, fitness_1)

    print("\nAfter adding alpha_1:")
    print("Contains alpha_1:", memory.contains(alpha_1))
    print("Fitness alpha_1 :", memory.get_fitness(alpha_1))

    print("\nTesting unseen alpha_2:")
    print("Contains alpha_2:", memory.contains(alpha_2))
    print("Fitness alpha_2 :", memory.get_fitness(alpha_2))

    print("\nMemory size:", memory.size())
    print("========================================")