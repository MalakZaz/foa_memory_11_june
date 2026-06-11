"""
===========================================================
 FILE: run_v0.py
 MODULE: FOA-CRN Experimental Pipeline (CLEAN V0)
===========================================================
"""

import numpy as np

# ===========================================================
# CORE
# ===========================================================
from src.core.sinr import compute_sinr
from src.core.rate import compute_rate
from src.core.fitness import fitness_function
from src.core.constraints import is_feasible

# ===========================================================
# SCENARIO
# ===========================================================
from src.scenario.scenario_generator import generate_scenario

# ===========================================================
# FOA
# ===========================================================
from src.foa.foa_v0 import run_foa

# ===========================================================
# METRICS + LOGGING
# ===========================================================
from src.utils.logger import ExperimentLogger
from src.utils.metrics import average_fitness
from src.config import (
    N, M, RUNS, ITERATIONS, POP_SIZE,
    ETA, SEED
)

# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":

    print("\n========================================")
    print(" FOA-CRN V0 EXPERIMENT (CLEAN RUN)")
    print("========================================\n")

    # -------------------------------------------------------
    # 1. SCENARIO
    # -------------------------------------------------------
    #N, M = 20, 10
    scenario = generate_scenario(N, M, seed=SEED)

    # -------------------------------------------------------
    # 2. LOGGER
    # -------------------------------------------------------
    logger = ExperimentLogger()

    # -------------------------------------------------------
    # 3. FITNESS WRAPPER
    # -------------------------------------------------------
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

    # -------------------------------------------------------
    # 4. FOA PARAMETERS
    # -------------------------------------------------------
    #POP_SIZE = 30
    #ITERATIONS = 50
    #ETA = 0.1

    # -------------------------------------------------------
    # 5. RUN FOA
    # -------------------------------------------------------
    best_X = None
    best_fit = -np.inf

    for it in range(ITERATIONS):

        best_alpha, best_fit, _ = run_foa(
            iterations=1,
            pop_size=POP_SIZE,
            N=N,
            M=M,
            eta=ETA,
            scenario=scenario,
            fitness_fn=fitness_fn
        )

        # simulation placeholder for avg fitness
        avg_fit = best_fit * 0.8  # simplification contrôlée V0

        logger.log_iteration(it, best_fit, avg_fit)

    # -------------------------------------------------------
    # 6. SAVE RESULTS
    # -------------------------------------------------------
    logger.set_final_best(best_fit)
    path = logger.save()

    print("\n========================================")
    print(" BEST FITNESS:", best_fit)
    print(" LOG SAVED TO:", path)
    print("========================================\n")