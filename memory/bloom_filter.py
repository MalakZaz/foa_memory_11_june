"""
bloom_filter.py - Bloom filter pour la détection de familiarité
"""

import hashlib
import numpy as np
from config import BLOOM_SIZE, BLOOM_HASHES


class BloomFilter:
    def __init__(self, m=None, k=None):
        self.m = m if m is not None else BLOOM_SIZE
        self.k = k if k is not None else BLOOM_HASHES
        self.bits = np.zeros(self.m, dtype=np.uint8)
    
    def _hash(self, element, seed):
        h = hashlib.sha256()
        h.update(seed.to_bytes(8, 'big'))
        h.update(element)
        return int.from_bytes(h.digest()[:8], 'big') % self.m
    
    def _flatten(self, alpha):
        return alpha.flatten().tobytes()
    
    def hash_allocation(self, alpha):
        flat = self._flatten(alpha)
        return [self._hash(flat, i) for i in range(self.k)]
    
    def add(self, indices):
        for idx in indices:
            self.bits[idx] = 1
    
    def contains(self, indices):
        return all(self.bits[idx] == 1 for idx in indices)
    
    def get_familiarity(self, indices):
        if not indices:
            return 0.0
        actifs = sum(1 for idx in indices if self.bits[idx] == 1)
        return actifs / len(indices)
    
    def clear(self):
        self.bits[:] = 0