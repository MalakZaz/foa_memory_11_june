"""
avoidance.py - Gestionnaire d'évitement
"""

from config import BETA_BAD, NOVELTY_THRESHOLD
from memory.bloom_filter import BloomFilter


class AvoidanceManager:
    def __init__(self, m=None, k=None, beta=None, novelty_threshold=None):
        from config import BLOOM_SIZE, BLOOM_HASHES
        self.bloom = BloomFilter(
            m=m if m is not None else BLOOM_SIZE,
            k=k if k is not None else BLOOM_HASHES
        )
        self.beta = beta if beta is not None else BETA_BAD
        self.novelty_threshold = novelty_threshold if novelty_threshold is not None else NOVELTY_THRESHOLD
        self.avoided_count = 0
        self.evaluated_count = 0
    
    def should_avoid(self, fitness, best_fitness, indices):
        if indices is None:
            return False
        
        familiarity = self.bloom.get_familiarity(indices)
        is_bad = fitness < self.beta * best_fitness
        
        if familiarity > self.novelty_threshold and is_bad:
            self.avoided_count += 1
            return True
        
        self.evaluated_count += 1
        return False
    
    def record_evaluation(self, indices, fitness, best_fitness):
        if indices is None:
            return
        is_bad = fitness < self.beta * best_fitness
        if is_bad:
            self.bloom.add(indices)
    
    def get_stats(self):
        total = self.avoided_count + self.evaluated_count
        return {
            'avoided_count': self.avoided_count,
            'evaluated_count': self.evaluated_count,
            'avoidance_rate': self.avoided_count / total if total > 0 else 0.0
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"  Évitements: {stats['avoided_count']}")
        print(f"  Évaluations: {stats['evaluated_count']}")
        print(f"  Taux d'évitement: {stats['avoidance_rate']*100:.1f}%")