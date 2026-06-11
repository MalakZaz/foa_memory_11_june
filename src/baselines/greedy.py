import numpy as np

from src.core.fitness import fitness_function
from src.core.repair import repair_solution
from src.config import USE_QOS

def run_greedy(N, M, scenario):

    h = scenario["h"]

    alpha = np.zeros((N, M), dtype=int)

    # assign best channel per SU
    for i in range(N):
        best_p = np.argmax(h[i])
        alpha[i, best_p] = 1

    # MKP + QoS repair
    alpha = repair_solution(
        alpha,
        scenario["b"],
        scenario["B"],
        h=scenario["h"],
        P_TX=scenario["P_TX"],
        SIGMA2=scenario["SIGMA2"],
        use_qos=USE_QOS
    )

    fit = fitness_function(
        alpha,
        h=scenario["h"],
        B=scenario["B"],
        P_TX=scenario["P_TX"],
        SIGMA2=scenario["SIGMA2"],
        w=scenario["w"],
        d=scenario["d"]
    )

    return alpha, fit