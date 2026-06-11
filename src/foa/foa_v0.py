"""
===========================================================
 FILE: foa_v0.py
 MODULE: Fruit Fly Optimization Algorithm (FOA V0)
===========================================================
"""

import numpy as np

from src.foa.population import initialize_population
from src.foa.movement import update
from src.core.repair import repair_solution
from src.config import USE_QOS


def binarize(X):
    """
    Convert continuous FOA position into MKP binary decision.
    """
    return (X > 0.5).astype(int)


def run_foa(iterations,
            pop_size,
            N,
            M,
            eta,
            scenario,
            fitness_fn):
    """
    FOA optimizer for MKP-based CRN spectrum allocation.
    """

    X = initialize_population(pop_size, N, M)

    best_X = X[0].copy()
    best_alpha = binarize(best_X)
    best_fit = -np.inf

    best_history = []

    for iteration in range(iterations):

        for i in range(pop_size):

            alpha = binarize(X[i])

            # MKP + optional QoS repair
            alpha = repair_solution(
                alpha,
                scenario["b"],
                scenario["B"],
                h=scenario["h"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                use_qos=USE_QOS
            )

            fit = fitness_fn(
                alpha,
                h=scenario["h"],
                B=scenario["B"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                w=scenario["w"],
                d=scenario["d"]
            )

            if fit > best_fit:
                best_fit = fit
                best_X = X[i].copy()
                best_alpha = alpha.copy()

        best_history.append(best_fit)

        X = update(X, best_X, eta)

    return best_alpha, best_fit, best_history