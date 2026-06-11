"""
qos.py - Vérification des contraintes QoS
"""

import numpy as np
from config import SINR_MIN, R_MIN, B_p
from core.sinr import compute_sinr


def check_qos(s, p, alpha, O):
    """
    Vérifie les contraintes QoS pour SU s sur canal p.
    Retourne (ok, sinr, rate)
    """
    sinr = compute_sinr(s, p, alpha, O)
    
    if sinr < SINR_MIN:
        return False, sinr, 0.0
    
    rate = B_p[p] * np.log2(1 + sinr)
    if rate < R_MIN:
        return False, sinr, rate
    
    return True, sinr, rate