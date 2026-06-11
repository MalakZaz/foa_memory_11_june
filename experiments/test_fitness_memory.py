import numpy as np

from src.memory.fitness_memory import FitnessMemory


memory = FitnessMemory()

alpha = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

print("Contains before:", memory.contains(alpha))

memory.add(alpha, 100.0)

print("Contains after :", memory.contains(alpha))
print("Stored fitness :", memory.get_fitness(alpha))

memory.add(alpha, 120.0)

print("Updated fitness:", memory.get_fitness(alpha))
print("Memory size    :", memory.size())