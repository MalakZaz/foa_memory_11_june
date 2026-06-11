"""
===========================================================
 FILE: rate.py
 MODULE: Spectral Efficiency / Rate Computation
 FOA-CRN V0
===========================================================

DESCRIPTION
-----------
Computes achievable data rate using Shannon capacity law.

This module transforms physical layer SINR into
optimization utility metric for MKP formulation.

===========================================================
MATHEMATICAL MODEL
-----------------------------------------------------------
R_{s,p} = B_p * log2(1 + SINR_{s,p})
===========================================================
"""

import numpy as np


# ===========================================================
# RATE COMPUTATION
# ===========================================================
def compute_rate(sinr, B):
    """
    Compute transmission rate for each SU on each channel.

    Parameters
    ----------
    sinr : (N,M) SINR matrix
    B    : (M,) channel bandwidths

    Returns
    -------
    rate : (N,M) achievable data rate
    """

    N, M = sinr.shape
    rate = np.zeros((N, M))

    for s in range(N):
        for p in range(M):

            if sinr[s, p] <= 0:
                continue

            # Shannon capacity formula
            rate[s, p] = B[p] * np.log2(1 + sinr[s, p])

    return rate


# ===========================================================
# QoS CHECK (RATE LEVEL)
# ===========================================================
def check_rate_qos(rate, R_min):
    """
    Verify minimum throughput constraint:
        R ≥ R_min
    """

    return np.all(rate[rate > 0] >= R_min)