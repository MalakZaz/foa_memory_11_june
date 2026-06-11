"""
===========================================================
 FILE: gwo_v0.py
 MODULE: Grey Wolf Optimizer (GWO) Baseline
===========================================================

Binary GWO for MKP-based spectrum allocation.
===========================================================
"""

import numpy as np

from src.core.repair import repair_solution

from src.config import USE_QOS
def binarize(position):
    """
    Convert continuous positions to binary.
    """
    return (position > 0.5).astype(int)


def run_gwo(iterations,
            pop_size,
            N,
            M,
            scenario,
            fitness_fn):

    # =======================================================
    # Initialize wolves
    # =======================================================
    wolves = np.random.rand(pop_size, N, M)

    alpha_pos = np.zeros((N, M))
    beta_pos = np.zeros((N, M))
    delta_pos = np.zeros((N, M))

    alpha_score = -np.inf
    beta_score = -np.inf
    delta_score = -np.inf

    history = []

    # =======================================================
    # Main loop
    # =======================================================
    for iteration in range(iterations):

        # ---------------------------------------------------
        # Evaluate wolves
        # ---------------------------------------------------
        for i in range(pop_size):

            alpha = binarize(wolves[i])

            # MKP + QoS repair
            alpha = repair_solution(
                alpha,
                scenario["b"],
                scenario["B"],
                h=scenario["h"],
                P_TX=scenario["P_TX"],
                SIGMA2=scenario["SIGMA2"],
                use_qos=USE_QOS
            )

            fitness = fitness_fn(alpha, **scenario)

            if fitness > alpha_score:

                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = alpha_score
                beta_pos = alpha_pos.copy()

                alpha_score = fitness
                alpha_pos = wolves[i].copy()

            elif fitness > beta_score:

                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = fitness
                beta_pos = wolves[i].copy()

            elif fitness > delta_score:

                delta_score = fitness
                delta_pos = wolves[i].copy()

        history.append(alpha_score)

        # ---------------------------------------------------
        # Update positions
        # ---------------------------------------------------
        a = 2 - iteration * (2 / iterations)

        for i in range(pop_size):

            r1 = np.random.rand(N, M)
            r2 = np.random.rand(N, M)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = np.abs(C1 * alpha_pos - wolves[i])
            X1 = alpha_pos - A1 * D_alpha

            r1 = np.random.rand(N, M)
            r2 = np.random.rand(N, M)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta_pos - wolves[i])
            X2 = beta_pos - A2 * D_beta

            r1 = np.random.rand(N, M)
            r2 = np.random.rand(N, M)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta_pos - wolves[i])
            X3 = delta_pos - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3

        wolves = np.clip(wolves, 0, 1)

    # =======================================================
    # Final solution
    # =======================================================
    best_alpha = binarize(alpha_pos)

    best_alpha = repair_solution(
        best_alpha,
        scenario["b"],
        scenario["B"],
        h=scenario["h"],
        P_TX=scenario["P_TX"],
        SIGMA2=scenario["SIGMA2"],
        use_qos=True
    )

    return best_alpha, alpha_score, history