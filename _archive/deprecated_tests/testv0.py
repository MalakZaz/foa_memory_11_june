from src.core.sinr import compute_sinr
import numpy as np

N, M = 3, 2
alpha = np.random.randint(0, 2, (N, M))
h = np.random.rand(N, M)

P_TX = 1.0
sigma2 = 0.1

sinr = compute_sinr(alpha, h, P_TX, sigma2)

print(sinr)