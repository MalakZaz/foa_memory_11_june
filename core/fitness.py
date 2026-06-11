"""
fitness.py - Fonction objectif (version tolérante)
"""

import numpy as np
from config import N, w_s, d_s, DELTA_T, SINR_MIN
from core.qos import check_qos


def compute_fitness(alpha, O):
    """
    Calcule la fitness.
    Les SUs qui ne respectent pas la QoS sont ignorés (pas de pénalité -inf).
    """
    total = 0.0
    nb_valid = 0
    
    for s in range(N):
        for p in range(alpha.shape[1]):
            if alpha[s, p] == 1:
                ok, sinr, rate = check_qos(s, p, alpha, O)
                if ok:
                    term = w_s[s] * min(d_s[s], DELTA_T) * rate * (sinr / SINR_MIN)
                    total += term
                    nb_valid += 1
    
    if nb_valid == 0:
        return 0.0
    
    return total / N