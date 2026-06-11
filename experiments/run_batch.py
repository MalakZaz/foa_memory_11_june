"""
===========================================================
 FILE: run_batch.py
 MODULE: FOA-CRN Multi-Run Experimental Framework
===========================================================

PURPOSE
-------
Runs FOA multiple times to ensure:

    - statistical validity
    - robustness analysis
    - convergence stability
    - reproducibility

Outputs:
    - mean fitness
    - std fitness
    - best/worst performance
    - convergence statistics

===========================================================
"""

import numpy as np

from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function
from src.foa.foa_v0 import run_foa


# ===========================================================
# SINGLE RUN WRAPPER
# ===========================================================
def run_single_experiment(N, M, scenario, pop_size, iterations, eta):
    """
    Executes one FOA run.
    """

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

    best_alpha, best_fit, history = run_foa(
        iterations=iterations,
        pop_size=pop_size,
        N=N,
        M=M,
        eta=eta,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    return best_fit


# ===========================================================
# MULTI-RUN EXPERIMENT
# ===========================================================
def run_batch(n_runs=20,
              N=20,
              M=10,
              pop_size=30,
              iterations=50,
              eta=0.1,
              seed=42):
    """
    Perform multiple FOA executions.
    """

    np.random.seed(seed)

    results = []

    for r in range(n_runs):

        scenario = generate_scenario(N, M, seed=seed + r)

        best_fit = run_single_experiment(
            N, M,
            scenario,
            pop_size,
            iterations,
            eta
        )

        results.append(best_fit)

        print(f"Run {r+1}/{n_runs} → Best Fitness = {best_fit}")

    results = np.array(results)

    summary = {
        "mean_fitness": float(np.mean(results)),
        "std_fitness": float(np.std(results)),
        "best_run": float(np.max(results)),
        "worst_run": float(np.min(results)),
        "all_results": results.tolist()
    }

    return summary


# ===========================================================
# ENTRY POINT
# ===========================================================
if __name__ == "__main__":

    print("\n========================================")
    print(" FOA-CRN MULTI-RUN BATCH EXPERIMENT")
    print("========================================\n")

    summary = run_batch(
        n_runs=20,
        N=20,
        M=10,
        pop_size=30,
        iterations=50,
        eta=0.1
    )

    print("\n========================================")
    print(" BATCH RESULTS SUMMARY")
    print("========================================\n")

    print("MEAN FITNESS :", summary["mean_fitness"])
    print("STD FITNESS  :", summary["std_fitness"])
    print("BEST RUN     :", summary["best_run"])
    print("WORST RUN    :", summary["worst_run"])

    print("\n========================================")