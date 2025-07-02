import numpy as np

def minkowski_distance(x1, x2, p_minkowski=2):
    return np.sum(np.abs(x1 - x2) ** p_minkowski) ** (1 / p_minkowski)
