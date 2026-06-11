"""
===========================================================
 FILE: solution_memory.py
 MODULE: FOA-Memory V1
===========================================================

PURPOSE
-------
This module stores previously explored allocation patterns
in order to reduce redundant evaluations during FOA search.

V1 implements a simple exact-memory mechanism.
Bloom-filter novelty detection will be added in V2.
===========================================================
"""

import numpy as np


# ===========================================================
# SOLUTION SIGNATURE
# ===========================================================
def solution_signature(alpha):
    """
    Convert a binary allocation matrix into a compact signature.

    Parameters
    ----------
    alpha : ndarray (N, M)
        Binary allocation matrix.

    Returns
    -------
    tuple
        Hashable representation of the allocation.
    """

    return tuple(alpha.astype(int).flatten())


# ===========================================================
# MEMORY CLASS
# ===========================================================
class SolutionMemory:
    """
    Stores signatures of previously evaluated solutions.
    """

    def __init__(self):
        self.visited = set()

    def contains(self, alpha):
        """
        Check whether alpha was already visited.
        """

        key = solution_signature(alpha)
        return key in self.visited

    def add(self, alpha):
        """
        Store alpha in memory.
        """

        key = solution_signature(alpha)
        self.visited.add(key)

    def size(self):
        """
        Return number of stored solutions.
        """

        return len(self.visited)