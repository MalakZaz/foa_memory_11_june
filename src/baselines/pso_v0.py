import numpy as np

from src.core.repair import repair_solution
from src.config import USE_QOS

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_pso(iterations,
            pop_size,
            N,
            M,
            scenario,
            fitness_fn):

    X = np.random.randn(pop_size, N, M)
    V = np.zeros((pop_size, N, M))

    pbest = X.copy()
    pbest_fit = np.full(pop_size, -np.inf)

    gbest = None
    gbest_fit = -np.inf

    history = []

    for _ in range(iterations):

        for i in range(pop_size):

            alpha = (X[i] > 0.5).astype(int)

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

            fit = fitness_fn(alpha)

            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = X[i].copy()

            if fit > gbest_fit:
                gbest_fit = fit
                gbest = X[i].copy()

        history.append(gbest_fit)

        for i in range(pop_size):

            r1 = np.random.rand(N, M)
            r2 = np.random.rand(N, M)

            V[i] = (
                0.7 * V[i]
                + 1.5 * r1 * (pbest[i] - X[i])
                + 1.5 * r2 * (gbest - X[i])
            )

            X[i] = sigmoid(V[i])

    best_alpha = (gbest > 0.5).astype(int)

    best_alpha = repair_solution(
        best_alpha,
        scenario["b"],
        scenario["B"],
        h=scenario["h"],
        P_TX=scenario["P_TX"],
        SIGMA2=scenario["SIGMA2"],
        use_qos=True
    )
    gbest_fit = fitness_fn(best_alpha)
    return best_alpha, gbest_fit, history