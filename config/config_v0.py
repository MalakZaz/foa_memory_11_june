# ============================================================
# FILE: config/config_v0.py
# PROJECT: FOA-based CRN Spectrum Allocation (v0 baseline)
# DESCRIPTION:
#   Centralized configuration for MKP-based spectrum allocation
#   using Fruit Fly Optimization Algorithm (FOA).
# ============================================================

import numpy as np

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
np.random.seed(SEED)

# ============================================================
# SYSTEM DIMENSIONS (MKP STRUCTURE)
# ============================================================
N_SUS = 20   # Number of Secondary Users (items)
M_CH = 10    # Number of channels (knapsacks)

# ============================================================
# PHYSICAL LAYER PARAMETERS
# ============================================================
P_TX = 1.0        # Transmission power (W)
SIGMA2 = 0.1      # Noise power

# ============================================================
# QUALITY OF SERVICE (QoS) CONSTRAINTS
# ============================================================
SINR_MIN = 2.0    # Minimum SINR threshold
R_MIN = 1.0       # Minimum data rate (Mbps)

# ============================================================
# TEMPORAL MODEL
# ============================================================
DELTA_T = 5       # Time slot duration

# ============================================================
# FOA PARAMETERS
# ============================================================
POP_SIZE = 30
ITERATIONS = 100
ETA = 0.1         # movement intensity (exploration factor)

# ============================================================
# INTERFERENCE MODEL (OPTIONAL EXTENSION V0)
# ============================================================
USE_INTERFERENCE_MATRIX = True