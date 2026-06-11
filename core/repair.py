"""
repair.py - Réparation des allocations invalides
"""

import numpy as np
from config import N, M, A, w_s, d_s, DELTA_T, B_p
from core.sinr import compute_sinr
from core.constraints import check_constraints


def compute_score(s, p, alpha, O):
    """Score = R * SINR / w_s"""
    sinr = compute_sinr(s, p, alpha, O)
    rate = B_p[p] * np.log2(1 + sinr)
    return rate * sinr / w_s[s]


def repair_allocation(alpha, O):
    """
    Répare une allocation invalide.
    Garde au plus 1 canal par SU, basé sur le score.
    """
    alpha_repaired = np.zeros((N, M), dtype=int)
    
    # Collecter toutes les allocations avec leur score
    allocations = []
    for s in range(N):
        for p in range(M):
            if alpha[s, p] == 1 and A[s, p] == 1:
                score = compute_score(s, p, alpha, O)
                allocations.append((score, s, p))
    
    # Trier par score décroissant
    allocations.sort(key=lambda x: x[0], reverse=True)
    
    # Ajouter une par une
    for score, s, p in allocations:
        if np.sum(alpha_repaired[s, :]) == 0:
            alpha_repaired[s, p] = 1
    
    return alpha_repaired