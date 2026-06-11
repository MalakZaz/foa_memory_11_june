"""
===========================================================
 FILE: repair.py
 MODULE: MKP Feasibility Repair Operator with QoS Support
===========================================================
"""

import numpy as np

from src.core.sinr import compute_sinr
from src.core.rate import compute_rate
from src.config import SINR_MIN, R_MIN


# ===========================================================
# EXCLUSIVITY FIX
# ===========================================================
def enforce_exclusivity(alpha):
    N, M = alpha.shape

    for i in range(N):
        if np.sum(alpha[i]) > 1:
            chosen = np.random.choice(np.where(alpha[i] == 1)[0])
            alpha[i, :] = 0
            alpha[i, chosen] = 1

    return alpha


# ===========================================================
# BANDWIDTH FIX
# ===========================================================
def enforce_bandwidth(alpha, b, B):
    N, M = alpha.shape

    for p in range(M):

        while np.sum(alpha[:, p] * b) > B[p]:

            candidates = np.where(alpha[:, p] == 1)[0]

            if len(candidates) == 0:
                break

            remove = np.random.choice(candidates)
            alpha[remove, p] = 0

    return alpha


# ===========================================================
# QoS FIX
# ===========================================================
def enforce_qos(alpha, h, B, P_TX, SIGMA2):
    """
    Remove allocations that violate QoS constraints:
        SINR >= SINR_MIN
        Rate >= R_MIN
    """

    sinr = compute_sinr(alpha, h, P_TX, SIGMA2)
    rate = compute_rate(sinr, B)

    N, M = alpha.shape

    for s in range(N):
        for p in range(M):

            if alpha[s, p] == 0:
                continue

            if sinr[s, p] < SINR_MIN:
                alpha[s, p] = 0
                continue

            if rate[s, p] < R_MIN:
                alpha[s, p] = 0
                continue

    return alpha


# ===========================================================
# MAIN EXPORT FUNCTION
# ===========================================================
def repair_solution(alpha,
                    b,
                    B,
                    h=None,
                    P_TX=None,
                    SIGMA2=None,
                    use_qos=False):
    """
    Full MKP repair pipeline.

    Steps:
        1. Enforce one-channel-per-SU exclusivity.
        2. Enforce channel bandwidth feasibility.
        3. Optionally enforce QoS constraints using config.py:
            - SINR_MIN
            - R_MIN
    """

    alpha = enforce_exclusivity(alpha.copy())
    alpha = enforce_bandwidth(alpha, b, B)

    if use_qos:
        #print("QoS constraints enabled")
        if h is None or P_TX is None or SIGMA2 is None:
            raise ValueError(
                "QoS repair requires h, P_TX, and SIGMA2."
            )

        alpha = enforce_qos(
            alpha=alpha,
            h=h,
            B=B,
            P_TX=P_TX,
            SIGMA2=SIGMA2
        )

    return alpha