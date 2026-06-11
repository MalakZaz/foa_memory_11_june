"""
===========================================================
 FILE: run_comparative.py
 MODULE: FOA vs Baselines Experimental Study
===========================================================
"""

import numpy as np

from src.scenario.scenario_generator import generate_scenario
from src.core.fitness import fitness_function
from src.foa.foa_v0 import run_foa
from src.baselines.comparator import run_baselines


# ===========================================================
# FOA WRAPPER
# ===========================================================
def run_foa_only(scenario, N, M):

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

    best_X, best_fit = run_foa(
        iterations=50,
        pop_size=30,
        N=N,
        M=M,
        eta=0.1,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    return best_fit


# ===========================================================
# MAIN COMPARATIVE STUDY
# ===========================================================
if __name__ == "__main__":

    print("\n========================================")
    print(" FOA-CRN COMPARATIVE STUDY")
    print("========================================\n")

    N, M = 20, 10
    scenario = generate_scenario(N, M, seed=42)

    # -------------------------------------------------------
    # FOA
    # -------------------------------------------------------
    foa_fit = run_foa_only(scenario, N, M)

    # -------------------------------------------------------
    # BASELINES
    # -------------------------------------------------------
    baseline_results = run_baselines(
        scenario=scenario,
        N=N,
        M=M
    )

    # -------------------------------------------------------
    # RESULTS TABLE
    # -------------------------------------------------------
    results = {
        "FOA": foa_fit,
        **baseline_results
    }

    print("\n========== COMPARATIVE RESULTS ==========\n")

    for k, v in results.items():
        print(f"{k:10s} : {v:.4f}")

    print("\n========================================\n")

    # -------------------------------------------------------
    # SIMPLE SCIENTIFIC TABLE EXPORT
    # -------------------------------------------------------
    import pandas as pd

    df = pd.DataFrame([results])
    df.to_csv("results/v0/comparative_table.csv", index=False)

    print("Table saved to results/v0/comparative_table.csv")