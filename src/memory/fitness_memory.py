"""
===========================================================
 FILE: fitness_memory.py
 MODULE: Fitness-Aware Memory for FOA-Memory V2
===========================================================

PURPOSE
-------
This module stores the best known fitness value associated
with each previously evaluated allocation pattern.

Unlike FOA-Memory V1, this memory does not blindly reject
all repeated solutions. It helps identify familiar bad
solutions only.
===========================================================
"""

from src.memory.solution_memory import solution_signature


class FitnessMemory:
    """
    Stores fitness values of previously evaluated solutions.
    """

    def __init__(self):
        self.memory = {}

    def contains(self, alpha):
        """
        Check whether alpha has already been evaluated.
        """

        key = solution_signature(alpha)
        return key in self.memory

    def get_fitness(self, alpha):
        """
        Return stored fitness of alpha.
        """

        key = solution_signature(alpha)
        return self.memory.get(key, None)

    def add(self, alpha, fitness):
        """
        Store or update fitness value of alpha.
        """

        key = solution_signature(alpha)

        if key not in self.memory:
            self.memory[key] = fitness
        else:
            self.memory[key] = max(self.memory[key], fitness)

    def size(self):
        """
        Number of remembered solutions.
        """

        return len(self.memory)