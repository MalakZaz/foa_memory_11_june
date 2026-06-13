"""
===========================================================
 FILE: bloom_filter.py
 MODULE: Bloom Filter for solution familiarity detection
===========================================================

PURPOSE
-------
Detect whether an allocation matrix has probably been seen
before.

The Bloom filter answers:
    "Is this solution familiar?"

It does not store fitness values.
===========================================================
"""

import hashlib
import numpy as np

from src.config import BLOOM_SIZE, BLOOM_HASHES


def solution_to_bytes(alpha):
    """
    Convert allocation matrix into bytes.
    """
    return alpha.astype(np.uint8).tobytes()


class BloomFilter:
    """
    Bloom Filter for binary allocation matrices.
    """

    def __init__(self,
                 size=BLOOM_SIZE,
                 num_hashes=BLOOM_HASHES):

        self.size = size
        self.num_hashes = num_hashes
        self.bits = np.zeros(size, dtype=bool)

    def _hashes(self, alpha):
        """
        Generate hash indices for an allocation matrix.
        """

        data = solution_to_bytes(alpha)

        indices = []

        for i in range(self.num_hashes):

            digest = hashlib.sha256(
                data + str(i).encode()
            ).hexdigest()

            index = int(digest, 16) % self.size

            indices.append(index)

        return indices

    def add(self, alpha):
        """
        Add a solution to the Bloom filter.
        """

        for index in self._hashes(alpha):
            self.bits[index] = True

    def contains(self, alpha):
        """
        Check whether a solution is probably familiar.
        """

        return all(
            self.bits[index]
            for index in self._hashes(alpha)
        )

    def familiarity(self, alpha):
        """
        Return the proportion of active hash bits.

        0.0 means completely novel.
        1.0 means probably familiar.
        """

        indices = self._hashes(alpha)

        active = sum(
            self.bits[index]
            for index in indices
        )

        return active / self.num_hashes

    def occupancy(self):
        """
        Return the percentage of activated Bloom bits.
        """

        return np.mean(self.bits)

    def reset(self):
        """
        Reset the Bloom filter.
        """

        self.bits[:] = False