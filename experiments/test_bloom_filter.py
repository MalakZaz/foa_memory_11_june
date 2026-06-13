"""
===========================================================
 FILE: test_bloom_filter.py
 PURPOSE: Test Bloom Filter familiarity detection
===========================================================
"""

import numpy as np

from src.memory.bloom_filter import BloomFilter
from src.config import N, M, BLOOM_SIZE, BLOOM_HASHES


if __name__ == "__main__":

    bloom = BloomFilter()

    alpha_1 = np.zeros((N, M), dtype=int)
    alpha_1[0, 0] = 1
    alpha_1[1, 1] = 1

    alpha_2 = np.zeros((N, M), dtype=int)
    alpha_2[2, 2] = 1

    print("\n========================================")
    print(" BLOOM FILTER TEST")
    print("========================================")
    print(f"Bloom size      : {BLOOM_SIZE}")
    print(f"Hash functions  : {BLOOM_HASHES}")
    print("----------------------------------------")

    print("Before adding alpha_1:")
    print("Contains alpha_1:", bloom.contains(alpha_1))
    print("Familiarity alpha_1:", bloom.familiarity(alpha_1))

    bloom.add(alpha_1)

    print("\nAfter adding alpha_1:")
    print("Contains alpha_1:", bloom.contains(alpha_1))
    print("Familiarity alpha_1:", bloom.familiarity(alpha_1))

    print("\nTesting unseen alpha_2:")
    print("Contains alpha_2:", bloom.contains(alpha_2))
    print("Familiarity alpha_2:", bloom.familiarity(alpha_2))

    print("\nBloom occupancy:", bloom.occupancy())

    print("========================================")