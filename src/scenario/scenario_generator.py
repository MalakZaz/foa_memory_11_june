import numpy as np


def generate_scenario(N, M, seed=42):

    np.random.seed(seed)

    # =====================================================
    # Physical Layer
    # =====================================================

    h = np.random.uniform(0.5, 2.0, (N, M))

    P_TX = 1.0
    SIGMA2 = 0.1
    SINR_MIN = 2.0

    # =====================================================
    # Channel capacities
    # =====================================================

    B = np.random.randint(10, 21, M)

    # =====================================================
    # SU requests
    # =====================================================

    b = np.random.randint(1, 6, N)

    d = np.random.randint(1, 6, N)

    t_start = np.random.randint(0, 10, N)

    t_end = t_start + d

    # =====================================================
    # PU availability windows
    # =====================================================

    t_pu_start = np.zeros(M)

    t_pu_end = np.random.randint(10, 20, M)

    # =====================================================
    # Utility weights
    # =====================================================

    w = np.ones(N)

    # =====================================================
    # Scenario dictionary
    # =====================================================

    return {
        "h": h,
        "B": B,
        "b": b,
        "d": d,
        "t_start": t_start,
        "t_end": t_end,
        "t_pu_start": t_pu_start,
        "t_pu_end": t_pu_end,
        "P_TX": P_TX,
        "SIGMA2": SIGMA2,
        "SINR_MIN": SINR_MIN,
        "w": w
    }