"""
===========================================================
 FILE: test_time_sensitive_bloom.py
 PURPOSE: Test time-sensitive Bloom filter
===========================================================
"""

import numpy as np

from src.memory.time_sensitive_bloom_filter import (
    TimeSensitiveBloomFilter
)

from src.config import (
    N,
    M
)


if __name__ == "__main__":

    bloom = TimeSensitiveBloomFilter()

    print("\nCONFIGURATION")
    print("-" * 40)

    print(f"Bloom size       : {bloom.size}")
    print(f"Hash functions   : {bloom.num_hashes}")
    print(f"Decay            : {bloom.decay}")
    print(f"Recovery         : {bloom.recovery}")

    print("-" * 40)

    alpha = np.zeros((N, M), dtype=int)

    alpha[0, 0] = 1
    alpha[1, 1] = 1

    print("\n========================================")
    print(" TIME-SENSITIVE BLOOM FILTER TEST")
    print("========================================")

    print("\nInitial state")
    print("Novelty      :", bloom.novelty_score(alpha))
    print("Familiarity  :", bloom.familiarity_score(alpha))
    print("Occupancy    :", bloom.occupancy())

    # First exposure
    bloom.add(alpha)

    print("\nAfter one exposure")
    print("Novelty      :", bloom.novelty_score(alpha))
    print("Familiarity  :", bloom.familiarity_score(alpha))
    print("Occupancy    :", bloom.occupancy())

    # Repeated exposures
    for _ in range(20):
        bloom.add(alpha)

    print("\nAfter repeated exposure")
    print("Novelty      :", bloom.novelty_score(alpha))
    print("Familiarity  :", bloom.familiarity_score(alpha))
    print("Occupancy    :", bloom.occupancy())

    # Recovery period
    for _ in range(50):
        bloom.recover()

    print("\nAfter recovery without exposure")
    print("Novelty      :", bloom.novelty_score(alpha))
    print("Familiarity  :", bloom.familiarity_score(alpha))
    print("Occupancy    :", bloom.occupancy())

    print("\n========================================")