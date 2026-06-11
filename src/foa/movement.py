import numpy as np

def update(X, Xbest, eta):
    """
    Classical FOA movement.

    Parameters
    ----------
    X : ndarray
        Current population

    Xbest : ndarray
        Best individual

    eta : float
        Exploration factor

    Returns
    -------
    Updated population
    """

    r1 = np.random.rand(*X.shape)
    r2 = np.random.rand(*X.shape)

    X_new = X * r1 + Xbest * (1 - r1) + eta * r2

    return np.clip(X_new, 0, 1)