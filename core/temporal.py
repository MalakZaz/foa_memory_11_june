"""
temporal.py - Calcul des chevauchements temporels
"""

import numpy as np
from config import t_start, d_s, N


def compute_overlap_matrix():
    """
    Calcule la matrice de chevauchement O[s][j].
    O[s][j] = 1 si les transmissions de s et j se chevauchent.
    """
    O = np.zeros((N, N), dtype=int)
    
    for s in range(N):
        start_s = t_start[s]
        end_s = start_s + d_s[s]
        
        for j in range(N):
            if s == j:
                continue
            start_j = t_start[j]
            end_j = start_j + d_s[j]
            
            if start_s < end_j and start_j < end_s:
                O[s, j] = 1
    
    return O