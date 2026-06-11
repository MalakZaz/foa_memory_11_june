"""
constraints.py - Vérification des contraintes C1 et C2
"""

import numpy as np
from config import N, M, A


def check_constraints(alpha, O=None, check_qos=False):
    """
    Vérifie les contraintes:
    C1: chaque SU a au plus 1 canal
    C2: alpha <= A
    C3: pas de collision (deux SUs qui se chevauchent sur même canal)
    C4 (optionnel): SINR >= SINR_MIN
    """
    from config import N, M, A, SINR_MIN
    
    # C1: un SU = un seul canal
    for s in range(N):
        if np.sum(alpha[s, :]) > 1:
            return False
    
    # C2: disponibilité
    for s in range(N):
        for p in range(M):
            if alpha[s, p] == 1 and A[s, p] == 0:
                return False
    
    # C3: pas de collision (utilise O si fourni)
    if O is not None:
        for p in range(M):
            # Trouver tous les SUs sur ce canal
            sus_on_p = [s for s in range(N) if alpha[s, p] == 1]
            # Vérifier les paires qui se chevauchent
            for i, s1 in enumerate(sus_on_p):
                for s2 in sus_on_p[i+1:]:
                    if O[s1, p] and O[s2, p]:
                        return False
    
    # C4: QoS (optionnel, plus coûteux)
    if check_qos:
        from core.qos import compute_SINR
        for s in range(N):
            for p in range(M):
                if alpha[s, p] == 1:
                    sinr = compute_SINR(s, p, alpha, O)
                    if sinr < SINR_MIN:
                        return False
    
    return True