"""
===========================================================
 FILE: config.py
 PURPOSE: Centralized experimental configuration
===========================================================
"""

import numpy as np

EXPERIMENT_NAME = "Scenario_01_N20_M10_QoS_ON"
# ===========================================================
# REPRODUCIBILITY
# ===========================================================
SEED = 42
np.random.seed(SEED)

# ===========================================================
# NETWORK PARAMETERS
# ===========================================================
N = 20                  # Number of Secondary Users (SUs)
M = 10                  # Number of Channels (PUs)

# ===========================================================
# CHANNEL PARAMETERS
# ===========================================================
P_TX = 1.0              # Transmission power
SIGMA2 = 0.1            # Noise power

# ===========================================================
# QoS PARAMETERS
# ===========================================================
SINR_MIN = 2.0          # Minimum SINR requirement
R_MIN = 1.0             # Minimum data rate (Mbps)

USE_QOS = True           # QoS enforcement

# ===========================================================
# TEMPORAL PARAMETERS
# ===========================================================
DELTA_T = 5             # Slot duration

# ===========================================================
# FOA PARAMETERS
# ===========================================================
POP_SIZE = 30
ITERATIONS = 50         # Consistent with experiments
ETA = 0.3

# ===========================================================
# EXPERIMENTAL PARAMETERS
# ===========================================================
RUNS = 30              # Number of independent runs

# ===========================================================
# MEMORY PARAMETERS
# ===========================================================
BAD_THRESHOLD = 0.95    # Selected threshold after sensitivity analysis

#==========================================================

#==========================================================

# ===========================================================
# GWO PARAMETERS
# ===========================================================

GWO_POP_SIZE = POP_SIZE
GWO_ITERATIONS = ITERATIONS

# ==========================================================
# BLOOM FILTER PARAMETERS
# ==========================================================
BLOOM_SIZE = 1000       # Number of bits in Bloom filter

BLOOM_HASHES = 5        # Number of hash functions

BLOOM_DECAY = 0.5       # aggressive decay for active traces

BLOOM_RECOVERY = 0.01  # slow recovery toward novelty

BLOOM_FAMILIARITY_THRESHOLD = 0.5

# ==========================================================
# NICHING PARAMETERS
# ==========================================================
NUM_NICHES = 3          # Number of sub-swarms