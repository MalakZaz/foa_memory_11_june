"""
===========================================================
 FILE: sinr.py
 MODULE: SINR Computation - FOA CRN V0
===========================================================

DESCRIPTION
-----------
Computes Signal-to-Interference-plus-Noise Ratio (SINR)
for each SU on each PU channel.

The SINR is a global metric depending on full allocation α.

===========================================================
MATHEMATICAL MODEL
-----------------------------------------------------------
SINR_{s,p} =
(P_TX * h_{s,p}) /
(σ^2 + Σ_{j≠s} α_{j,p} * P_TX * h_{j,p})
===========================================================
"""

import numpy as np


# ===========================================================
# SINR COMPUTATION
# ===========================================================
def compute_sinr(alpha, h, P_TX, sigma2):
    """
    Compute SINR matrix (N x M)

    Parameters
    ----------
    alpha : (N,M) binary allocation matrix
    h     : (N,M) channel gain matrix
    P_TX  : transmission power
    sigma2: noise power

    Returns
    -------
    sinr : (N,M) SINR values
    """

    N, M = alpha.shape
    sinr = np.zeros((N, M))

    # Loop over users and channels
    for s in range(N):
        for p in range(M):

            if alpha[s, p] == 0:
                continue  # no transmission → SINR irrelevant

            signal = P_TX * h[s, p]

            interference = 0.0

            # SU-SU interference on same channel
            for j in range(N):
                if j != s and alpha[j, p] == 1:
                    interference += P_TX * h[j, p]

            sinr[s, p] = signal / (sigma2 + interference)

    return sinr


# ===========================================================
# QoS CHECK (SINR LEVEL)
# ===========================================================
def check_sinr_qos(sinr, gamma_min):
    """
    Verify SINR feasibility constraint:
        SINR ≥ γ_min
    """

    return np.all(sinr[sinr > 0] >= gamma_min)

if __name__ == "__main__":

    N, M = 3, 2
    alpha = np.random.randint(0, 2, (N, M))
    h = np.random.rand(N, M)

    P_TX = 1.0
    sigma2 = 0.1

    sinr = compute_sinr(alpha, h, P_TX, sigma2)

    print("SINR TEST:")
    print(sinr)