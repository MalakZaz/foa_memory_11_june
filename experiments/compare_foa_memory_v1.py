"""
===========================================================
 FILE: compare_foa_memory_v1.py
 MODULE: FOA V0 vs FOA-Memory V1 Comparison
===========================================================
"""

import numpy as np

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.foa.foa_memory_v1 import run_foa_memory_v1


def build_fitness_fn(scenario):

    def fitness_fn(alpha, **_):
        return fitness_function(
            alpha,
            h=scenario["h"],
            B=scenario["B"],
            P_TX=scenario["P_TX"],
            SIGMA2=scenario["SIGMA2"],
            w=scenario["w"],
            d=scenario["d"]
        )

    return fitness_fn


if __name__ == "__main__":

    set_seed(42)

    N, M = 20, 10
    ITERATIONS = 50
    POP_SIZE = 30
    ETA = 0.1

    scenario = generate_scenario(N, M, seed=42)
    fitness_fn = build_fitness_fn(scenario)

    print("\n========================================")
    print(" FOA V0 vs FOA-Memory V1")
    print("========================================\n")

    _, fit_v0, hist_v0 = run_foa(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    _, fit_mem, hist_mem, skipped, mem_size = run_foa_memory_v1(
        iterations=ITERATIONS,
        pop_size=POP_SIZE,
        N=N,
        M=M,
        eta=ETA,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    improvement = ((fit_mem - fit_v0) / abs(fit_v0)) * 100
    avoidance_rate = (sum(skipped) / (ITERATIONS * POP_SIZE)) * 100

    print(f"FOA V0 best fitness       : {fit_v0:.4f}")
    print(f"FOA-Memory V1 best fitness: {fit_mem:.4f}")
    print(f"Improvement (%)           : {improvement:.2f}%")
    print(f"Total skipped evaluations : {sum(skipped)}")
    print(f"Avoidance rate (%)        : {avoidance_rate:.2f}%")
    print(f"Final memory size         : {mem_size[-1]}")

    print("\n========================================")