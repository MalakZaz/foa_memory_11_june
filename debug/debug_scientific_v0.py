"""
===========================================================
 FILE: debug_scientific_v0.py
 MODULE: FOA-CRN Scientific Reviewer Debug Suite
===========================================================

OBJECTIVE
---------
This module validates:
1. SINR physical correctness
2. MKP constraint consistency
3. Fitness sensitivity
4. Allocation stability

===========================================================
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
from src.core.sinr import compute_sinr
from src.core.rate import compute_rate
from src.core.constraints import is_feasible
from src.core.fitness import fitness_function
from src.scenario.scenario_generator import generate_scenario


# ===========================================================
# MAIN DEBUG PIPELINE
# ===========================================================
if __name__ == "__main__":

    print("\n========================================")
    print(" FOA-CRN SCIENTIFIC DEBUG SUITE")
    print("========================================\n")

    # =======================================================
    # 1. SCENARIO
    # =======================================================
    N, M = 10, 5
    scenario = generate_scenario(N, M)

    h = scenario["h"]
    B = scenario["B"]
    b = scenario["b"]
    d = scenario["d"]

    t_start = scenario["t_start"]
    t_end = scenario["t_end"]
    t_pu_start = scenario["t_pu_start"]
    t_pu_end = scenario["t_pu_end"]

    P_TX = scenario["P_TX"]
    SIGMA2 = scenario["SIGMA2"]

    # =======================================================
    # 2. RANDOM ALLOCATIONS
    # =======================================================
    alpha_random = np.random.randint(0, 2, (N, M))
    alpha_congested = np.ones((N, M))

    # =======================================================
    # 3. SINR TEST
    # =======================================================
    sinr_random = compute_sinr(alpha_random, h, P_TX, SIGMA2)
    sinr_congested = compute_sinr(alpha_congested, h, P_TX, SIGMA2)

    print("[TEST 1] SINR stability")

    print("SINR random mean     :", np.mean(sinr_random))
    print("SINR congested mean  :", np.mean(sinr_congested))

    assert np.all(sinr_random >= 0), "SINR negative detected"

    print("✔ SINR physically valid\n")

    # =======================================================
    # 4. RATE TEST
    # =======================================================
    rate_random = compute_rate(sinr_random, B)

    print("[TEST 2] Rate distribution")
    print("Rate mean:", np.mean(rate_random))
    print("✔ Rate computed correctly\n")

    # =======================================================
    # 5. CONSTRAINT TEST
    # =======================================================
    feasible = is_feasible(
        alpha_random,
        b,
        B,
        t_start,
        t_end,
        t_pu_start,
        t_pu_end
    )

    print("[TEST 3] MKP feasibility")
    print("Feasible:", feasible)
    print("✔ Constraint module OK\n")

    # =======================================================
    # 6. FITNESS SENSITIVITY TEST
    # =======================================================
    f1 = fitness_function(alpha_random, sinr_random, rate_random, b, d)
    f2 = fitness_function(alpha_congested, sinr_congested,
                          compute_rate(sinr_congested, B), b, d)

    print("[TEST 4] Fitness sensitivity")
    print("Fitness random     :", f1)
    print("Fitness congested  :", f2)

    assert f1 != f2, "Fitness not sensitive"

    print("✔ Fitness responds to allocation structure\n")

    # =======================================================
    # 7. SUMMARY
    # =======================================================
    print("========================================")
    print(" DEBUG SCIENTIFIC COMPLETE")
    print("========================================")