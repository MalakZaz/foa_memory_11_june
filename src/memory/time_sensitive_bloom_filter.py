"""
===========================================================
 FILE: time_sensitive_bloom_filter.py
 MODULE: Time-Sensitive Bloom Filter
===========================================================

PURPOSE
-------
Biologically inspired Bloom filter based on gradual forgetting.

Inspired by Dasgupta et al. (2018):
- active memory traces become familiar after exposure;
- familiarity decreases progressively over time;
- novelty is continuous rather than binary.

===========================================================
"""

import hashlib
import numpy as np

from src.config import (
    BLOOM_SIZE,
    BLOOM_HASHES,
    BLOOM_DECAY,
    BLOOM_RECOVERY
)


def solution_to_bytes(alpha):
    """
    Convert allocation matrix into bytes.
    """
    return alpha.astype(np.uint8).tobytes()


class TimeSensitiveBloomFilter:
    """
    Time-sensitive Bloom Filter.

    Interpretation:
        1.0 -> completely novel
        0.0 -> completely familiar
    """

    def __init__(self,
                 size=BLOOM_SIZE,
                 num_hashes=BLOOM_HASHES,
                 decay=BLOOM_DECAY,
                 recovery=BLOOM_RECOVERY):

        self.size = size
        self.num_hashes = num_hashes
        self.decay = decay
        self.recovery = recovery

        # Initially all memories are novel
        self.weights = np.ones(size, dtype=float)

    def _hashes(self, alpha):
        """
        Generate Bloom indices.
        """

        data = solution_to_bytes(alpha)

        indices = []

        for i in range(self.num_hashes):

            digest = hashlib.sha256(
                data + str(i).encode()
            ).hexdigest()

            indices.append(
                int(digest, 16) % self.size
            )

        return indices

    def novelty_score(self, alpha):
        """
        Novelty score ∈ [0,1].

        1.0 → novel
        0.0 → familiar
        """

        indices = self._hashes(alpha)

        return float(
            np.mean(self.weights[indices])
        )

    def familiarity_score(self, alpha):
        """
        Familiarity score ∈ [0,1].
        """

        return 1.0 - self.novelty_score(alpha)

    def is_familiar(self,
                    alpha,
                    threshold=0.5):
        """
        Determine if solution is familiar.
        """

        return (
            self.novelty_score(alpha)
            < threshold
        )

    def add(self, alpha):
        """
        Exposure effect.

        Familiarity increases.
        """

        indices = self._hashes(alpha)

        self.weights[indices] *= self.decay

    def recover(self):
        """
        Time effect.

        Novelty progressively recovers.
        """

        self.weights = np.minimum(
            1.0,
            self.weights + self.recovery
        )

    def occupancy(self):
        """
        Estimate memory saturation.
        """

        return float(
            1.0 - np.mean(self.weights)
        )

    def reset(self):
        """
        Full reset.

        Used only for debugging.
        """

        self.weights[:] = 1.0