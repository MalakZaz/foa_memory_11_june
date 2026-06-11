"""
===========================================================
 FILE: run_convergence_data.py
 MODULE: Generate convergence histories for paper figures
===========================================================

PURPOSE
-------
This script runs FOA, PSO, Random Search, and Greedy over
multiple independent runs and saves the full convergence
histories into a CSV file.

The plotting script will later read this CSV instead of
rerunning the algorithms.
===========================================================
"""

import os
import pandas as pd
import numpy as np

from src.utils.seed import set_seed
from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function

from src.foa.foa_v0 import run_foa
from src.baselines.pso_v0 import run_pso
from src.baselines.random_search import run_random_search
from src.baselines.greedy import run_greedy
from src.config import (
    N, M, RUNS, ITERATIONS, POP_SIZE,
    ETA, SEED, BAD_THRESHOLD
)

# ===========================================================
# EXPERIMENT PARAMETERS
# ===========================================================


SAVE_PATH = f"results/v0/convergence_histories_N{N}_M{M}_RUNS{RUNS}_ITERATIONS{ITERATIONS}.csv"


# ===========================================================
# FITNESS WRAPPER
# ===========================================================
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


# ===========================================================
# RANDOM HISTORY
# ===========================================================
def random_history(iterations, N, M, scenario):

    history = []
    best_so_far = -np.inf

    for _ in range(iterations):

        _, fit = run_random_search(
            iterations=1,
            N=N,
            M=M,
            scenario=scenario
        )

        best_so_far = max(best_so_far, fit)
        history.append(best_so_far)

    return history


# ===========================================================
# GREEDY HISTORY
# ===========================================================
def greedy_history(iterations, N, M, scenario):

    _, fit = run_greedy(N, M, scenario)

    return [fit] * iterations


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":

    os.makedirs("results/v0", exist_ok=True)

    rows = []

    print("\n========================================")
    print(" GENERATING CONVERGENCE DATA")
    print("========================================\n")

    print(f"Runs       : {RUNS}")
    print(f"Iterations : {ITERATIONS}")
    print(f"N          : {N}")
    print(f"M          : {M}")
    print(f"Population : {POP_SIZE}")
    print(f"ETA        : {ETA}")

    for run in range(RUNS):

        seed = SEED + run

        print(f"\nRun {run + 1}/{RUNS} | seed={seed}")

        scenario = generate_scenario(N, M, seed=seed)
        fitness_fn = build_fitness_fn(scenario)

        # ---------------------------------------------------
        # FOA
        # ---------------------------------------------------
        set_seed(seed)

        _, _, foa_hist = run_foa(
            iterations=ITERATIONS,
            pop_size=POP_SIZE,
            N=N,
            M=M,
            eta=ETA,
            scenario=scenario,
            fitness_fn=fitness_fn
        )

        # ---------------------------------------------------
        # PSO
        # ---------------------------------------------------
        set_seed(seed)

        _, _, pso_hist = run_pso(
            iterations=ITERATIONS,
            pop_size=POP_SIZE,
            N=N,
            M=M,
            scenario=scenario,
            fitness_fn=fitness_fn
        )

        # ---------------------------------------------------
        # Random Search
        # ---------------------------------------------------
        set_seed(seed)

        random_hist = random_history(
            iterations=ITERATIONS,
            N=N,
            M=M,
            scenario=scenario
        )

        # ---------------------------------------------------
        # Greedy
        # ---------------------------------------------------
        greedy_hist = greedy_history(
            iterations=ITERATIONS,
            N=N,
            M=M,
            scenario=scenario
        )

        # ---------------------------------------------------
        # Save rows
        # ---------------------------------------------------
        for iteration in range(ITERATIONS):

            rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "FOA",
                "best_fitness": foa_hist[iteration]
            })

            rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "PSO",
                "best_fitness": pso_hist[iteration]
            })

            rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "Random Search",
                "best_fitness": random_hist[iteration]
            })

            rows.append({
                "run": run + 1,
                "seed": seed,
                "iteration": iteration + 1,
                "algorithm": "Greedy",
                "best_fitness": greedy_hist[iteration]
            })

    df = pd.DataFrame(rows)

    df.to_csv(SAVE_PATH, index=False)

    print("\n========================================")
    print(" CONVERGENCE DATA SAVED")
    print("========================================")
    print(f"Saved to: {SAVE_PATH}")
    print(f"Rows    : {len(df)}")
    print("========================================\n")