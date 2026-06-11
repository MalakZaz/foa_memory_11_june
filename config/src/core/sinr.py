import numpy as np
from src.config import P_TX, SIGMA2

def compute_sinr(alpha, h, O):
    N, M = alpha.shape
    sinr = np.zeros((N, M))

    for s in range(N):
        for p in range(M):

            if alpha[s, p] == 0:
                continue

            signal = P_TX * h[s, p]

            interference = 0.0

            for j in range(N):
                if j != s and alpha[j, p] == 1:
                    interference += O[s, j] * P_TX * h[j, p]

            sinr[s, p] = signal / (SIGMA2 + interference)

    return sinr