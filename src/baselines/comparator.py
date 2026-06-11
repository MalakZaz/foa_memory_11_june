"""
===========================================================
 FILE: comparator.py
 MODULE: Multi-Algorithm Comparison Engine
===========================================================
"""

import numpy as np

from src.baselines.random_search import run_random_search
from src.baselines.greedy import run_greedy
from src.baselines.pso_v0 import run_pso


# ===========================================================
# RUN ALL BASELINES
# ===========================================================
def run_baselines(scenario,
                  N,
                  M,
                  iterations=50,
                  pop_size=30):
    """
    Executes all benchmark algorithms.
    """

    results = {}

    # -------------------------------------------------------
    # RANDOM
    # -------------------------------------------------------
    _, random_fit = run_random_search(
        iterations=iterations,
        N=N,
        M=M,
        scenario=scenario
    )
    results["random"] = random_fit

    # -------------------------------------------------------
    # GREEDY
    # -------------------------------------------------------
    _, greedy_fit = run_greedy(
        N=N,
        M=M,
        scenario=scenario
    )
    results["greedy"] = greedy_fit

    # -------------------------------------------------------
    # PSO
    # -------------------------------------------------------
    def fitness_fn(alpha, **_):
        from src.core.fitness import fitness_function

        return fitness_function(
            alpha,
            h=scenario["h"],
            B=scenario["B"],
            P_TX=scenario["P_TX"],
            SIGMA2=scenario["SIGMA2"],
            w=scenario["w"],
            d=scenario["d"]
        )

    _, pso_fit = run_pso(
        iterations=iterations,
        pop_size=pop_size,
        N=N,
        M=M,
        scenario=scenario,
        fitness_fn=fitness_fn
    )

    results["pso"] = pso_fit

    return results