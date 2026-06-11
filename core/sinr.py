"""
sinr.py - Calcul du SINR
"""

import numpy as np
from config import N, P_TX, NOISE_POWER, h


def compute_interference(s, p, alpha, O):
    """Calcule l'interférence sur SU s depuis les autres SUs."""
    interference = 0.0
    for j in range(N):
        if j == s:
            continue
        if alpha[j, p] == 1 and O[s, j] == 1:
            interference += P_TX * h[j, p]
    return interference


def compute_sinr(s, p, alpha, O):
    """Calcule le SINR pour SU s sur canal p."""
    signal = P_TX * h[s, p]
    interference = compute_interference(s, p, alpha, O)
    return signal / (NOISE_POWER + interference + 1e-10)