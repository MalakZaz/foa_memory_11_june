"""
config.py - Configuration FOA V0 (FROZEN) 02 juin 9h:22
"""

import numpy as np

# ============================================================
# DIMENSIONS
# ============================================================
N = 40
M = 10

# ============================================================
# PARAMÈTRES FOA
# ============================================================
POPULATION = 50
ITERATIONS = 100
ETA = 0.1

# ============================================================
# PARAMÈTRES QoS (FIXÉS)
# ============================================================
R_MIN = 1.0          # Mbps
SINR_MIN = 2.0       # ratio linéaire (~3 dB)

# ============================================================
# PARAMÈTRES MÉMOIRE BLOOM (V2)
# ============================================================
BLOOM_SIZE = 1000
BLOOM_HASHES = 5
BETA_BAD = 0.7
NOVELTY_THRESHOLD = 0.5

# ============================================================
# PARAMÈTRES NICHING (V1, V3)
# ============================================================
SIGMA_SHARE = 0.15
ALPHA_SHARE = 2.0
# ============================================================
# PARAMÈTRES RADIO
# ============================================================
P_TX = 1.0
NOISE_POWER = 0.1

# ============================================================
# REPRODUCTIBILITÉ
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# DONNÉES GÉNÉRÉES (FIXES)
# ============================================================

# SUs
b_s = np.random.randint(4, 10, N)
t_start = np.random.uniform(0, 20, N)
d_s = np.random.randint(2, 10, N)
w_s = np.ones(N)

# Canaux
B_p = np.random.randint(8, 15, M)
I_max = np.random.uniform(5, 10, M)
gamma = np.random.uniform(0.5, 3.0, (N, M))
A = np.random.choice([0, 1], size=(N, M), p=[0.3, 0.7])
h = np.random.uniform(0.5, 2.0, (N, M))

# Fenêtre temporelle (calculée dynamiquement)
DELTA_T = np.max(d_s)