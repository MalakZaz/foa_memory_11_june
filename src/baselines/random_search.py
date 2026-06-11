import numpy as np

from src.core.fitness import fitness_function
from src.core.repair import repair_solution
from src.config import USE_QOS

def run_random_search(iterations, N, M, scenario):

    best_fit = -np.inf
    best_alpha = None

    for _ in range(iterations):

        alpha = np.random.randint(0, 2, (N, M))

        # MKP repair
        #alpha = repair_solution(alpha, scenario["b"], scenario["B"])
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

        if fit > best_fit:
            best_fit = fit
            best_alpha = alpha.copy()

    return best_alpha, best_fit