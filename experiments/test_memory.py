import numpy as np

from src.memory.solution_memory import SolutionMemory


memory = SolutionMemory()

alpha = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

print("Before add:", memory.contains(alpha))

memory.add(alpha)

print("After add :", memory.contains(alpha))
print("Memory size:", memory.size())