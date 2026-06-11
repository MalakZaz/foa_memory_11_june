"""
===========================================================
 FILE: constraints.py
 MODULE: Constraint Handling - FOA CRN V0
 PROJECT: Cognitive Radio Network Optimization (MKP-FOA)
===========================================================

DESCRIPTION
-----------
This module enforces all feasibility constraints of the
Multiple Knapsack Problem (MKP) formulation for spectrum
allocation in Cognitive Radio Networks (CRN).

It ensures:
- PU protection
- bandwidth feasibility
- temporal feasibility
- exclusive allocation per SU

===========================================================
MATHEMATICAL CONSTRAINTS
-----------------------------------------------------------
1. Bandwidth:
   Σ_i b_i x_i,p ≤ B_p

2. Time window:
   t_i ∈ [t_p_start, t_p_end]

3. Exclusivity:
   Σ_p x_i,p ≤ 1

4. Admissibility:
   d_i ≤ Δt_p
===========================================================
"""

import numpy as np


# ===========================================================
# 1. ADMISSIBILITY CHECK
# ===========================================================
def compute_admissible_users(d, pu_durations):
    """
    Determine which SUs are admissible.

    A user is admissible if:
        d_i ≤ Δt_p for at least one channel p
    """

    N = len(d)
    M = len(pu_durations)

    admissible = np.zeros(N, dtype=int)

    for i in range(N):
        for p in range(M):
            if d[i] <= pu_durations[p]:
                admissible[i] = 1
                break

    return admissible


# ===========================================================
# 2. EXCLUSIVITY CONSTRAINT
# ===========================================================
def check_exclusivity(alpha):
    """
    Each SU can be assigned to at most one channel
    Σ_p x_i,p ≤ 1
    """

    return np.all(np.sum(alpha, axis=1) <= 1)


# ===========================================================
# 3. BANDWIDTH CONSTRAINT
# ===========================================================
def check_bandwidth(alpha, b, B):
    """
    Σ_i b_i x_i,p ≤ B_p
    """

    M = alpha.shape[1]

    for p in range(M):
        if np.sum(b * alpha[:, p]) > B[p]:
            return False

    return True


# ===========================================================
# 4. TEMPORAL FEASIBILITY
# ===========================================================
def check_temporal(alpha, t_start, t_end, pu_start, pu_end):
    """
    Ensures SU transmission fits PU availability window
    """

    N, M = alpha.shape

    for i in range(N):
        for p in range(M):

            if alpha[i, p] == 1:

                if t_start[i] < pu_start[p]:
                    return False

                if t_end[i] > pu_end[p]:
                    return False

    return True


# ===========================================================
# 5. MASTER VALIDATION FUNCTION
# ===========================================================
def is_feasible(alpha, b, B, t_start, t_end, pu_start, pu_end):
    """
    Global feasibility check for MKP allocation
    """

    if not check_exclusivity(alpha):
        return False

    if not check_bandwidth(alpha, b, B):
        return False

    if not check_temporal(alpha, t_start, t_end, pu_start, pu_end):
        return False

    return True