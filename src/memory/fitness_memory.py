"""
===========================================================
 FILE: fitness_memory.py
 MODULE: Exact fitness memory for allocation matrices
===========================================================
"""

from src.memory.solution_memory import solution_signature


class FitnessMemory:
    """
    Stores previously evaluated solutions and their fitness values.
    """

    def __init__(self):
        self.memory = {}

    def add(self, alpha, fitness):
        signature = solution_signature(alpha)
        self.memory[signature] = fitness

    def contains(self, alpha):
        signature = solution_signature(alpha)
        return signature in self.memory

    def get_fitness(self, alpha):
        signature = solution_signature(alpha)
        return self.memory.get(signature, None)

    def size(self):
        return len(self.memory)

    def reset(self):
        self.memory.clear()s